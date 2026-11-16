"""Tests for the reverse DNS (PTR) hostname resolver.

No network access: DNS answers are hand-built byte strings, lookups are
replaced by fake query functions and time is a controllable counter.
"""
import asyncio
import struct

import pytest

from ha_zyxel.const import DNS_NEGATIVE_CACHE_TTL, DNS_POSITIVE_CACHE_TTL
from ha_zyxel.config_flow import is_valid_dns_server
from ha_zyxel.dns_resolver import (
    DnsFormatError,
    DnsHostnameResolver,
    DnsLookupError,
    DnsServerError,
    build_ptr_query,
    chain_hostname_lookups,
    fill_missing_hostnames,
    normalize_hostname,
    parse_ptr_response,
    resolvable_ip,
    reverse_pointer_name,
)

TXID = 0x1A2B
QNAME = "10.2.0.192.in-addr.arpa"


# ---------------------------------------------------------------------------
# Wire format
# ---------------------------------------------------------------------------

def _encoded_qname() -> bytes:
    return b"\x0210\x012\x010\x03192\x07in-addr\x04arpa\x00"


def test_reverse_pointer_name():
    assert reverse_pointer_name("192.0.2.10") == QNAME
    assert reverse_pointer_name("2001:db8::1").endswith(".ip6.arpa")


def test_build_ptr_query_bytes():
    query = build_ptr_query(QNAME, TXID)
    expected = (
        b"\x1a\x2b"  # id
        b"\x01\x00"  # flags: RD
        b"\x00\x01\x00\x00\x00\x00\x00\x00"  # 1 question
        + _encoded_qname()
        + b"\x00\x0c\x00\x01"  # PTR, IN
    )
    assert query == expected


def _response(flags: int, answers: bytes = b"", ancount: int = 0) -> bytes:
    header = struct.pack(">HHHHHH", TXID, flags, 1, ancount, 0, 0)
    question = _encoded_qname() + b"\x00\x0c\x00\x01"
    return header + question + answers


def _ptr_answer_compressed() -> bytes:
    # Owner name: pointer to the question name at offset 12 (0xC00C).
    # RDATA "printer.example.lan.": "printer" as a label, then a pointer to
    # "example.lan" stored in an earlier record to exercise nested pointers.
    rdata = b"\x07printer\x07example\x03lan\x00"
    return b"\xc0\x0c" + struct.pack(">HHIH", 12, 1, 600, len(rdata)) + rdata


def test_parse_ptr_response_with_name_compression():
    data = _response(0x8180, _ptr_answer_compressed(), ancount=1)
    assert parse_ptr_response(data, TXID) == "printer.example.lan"


def test_parse_ptr_response_pointer_inside_rdata():
    # First answer: CNAME whose target contains "example.lan" at a known
    # offset; second answer: PTR "printer" + pointer to that "example.lan".
    base = _response(0x8180, b"", ancount=2)
    cname_rdata = b"\x04host\x07example\x03lan\x00"
    cname = b"\xc0\x0c" + struct.pack(">HHIH", 5, 1, 600, len(cname_rdata)) + cname_rdata
    example_offset = len(base) + 2 + 10 + 5  # skip pointer, RR header, "\x04host"
    ptr_rdata = b"\x07printer" + struct.pack(">H", 0xC000 | example_offset)
    ptr = b"\xc0\x0c" + struct.pack(">HHIH", 12, 1, 600, len(ptr_rdata)) + ptr_rdata
    assert parse_ptr_response(base + cname + ptr, TXID) == "printer.example.lan"


def test_parse_ptr_response_nxdomain_returns_none():
    assert parse_ptr_response(_response(0x8183), TXID) is None


def test_parse_ptr_response_no_answer_returns_none():
    assert parse_ptr_response(_response(0x8180), TXID) is None


def test_parse_ptr_response_servfail_raises():
    with pytest.raises(DnsServerError):
        parse_ptr_response(_response(0x8182), TXID)


def test_parse_ptr_response_rejects_bad_messages():
    good = _response(0x8180, _ptr_answer_compressed(), ancount=1)
    with pytest.raises(DnsFormatError):
        parse_ptr_response(good, TXID + 1)  # wrong id
    with pytest.raises(DnsFormatError):
        parse_ptr_response(good[:8], TXID)  # shorter than header
    with pytest.raises(DnsFormatError):
        parse_ptr_response(good[:-5], TXID)  # truncated rdata
    with pytest.raises(DnsFormatError):
        parse_ptr_response(_response(0x0100), TXID)  # query, not a response
    with pytest.raises(DnsLookupError):
        parse_ptr_response(_response(0x8380), TXID)  # TC bit


def test_parse_ptr_response_pointer_loop_raises():
    base = _response(0x8180, b"", ancount=1)
    loop_offset = len(base) + 2 + 10
    rdata = struct.pack(">H", 0xC000 | loop_offset)  # points to itself
    answer = b"\xc0\x0c" + struct.pack(">HHIH", 12, 1, 600, len(rdata)) + rdata
    with pytest.raises(DnsFormatError):
        parse_ptr_response(base + answer, TXID)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def test_normalize_hostname_strips_domain():
    assert normalize_hostname("printer.example.lan.", True) == "printer"
    assert normalize_hostname("printer.example.lan.", False) == "printer.example.lan"
    assert normalize_hostname("printer", True) == "printer"
    assert normalize_hostname("", True) is None
    assert normalize_hostname(None, True) is None
    assert normalize_hostname(".", True) is None
    # Some resolvers echo the IP back instead of failing.
    assert normalize_hostname("192.0.2.10", True) is None


def test_resolvable_ip_skips_unspecified_and_invalid():
    assert resolvable_ip("192.0.2.10") == "192.0.2.10"
    assert resolvable_ip("0.0.0.0") is None
    assert resolvable_ip("") is None
    assert resolvable_ip(None) is None
    assert resolvable_ip("Unknown") is None


def test_is_valid_dns_server():
    assert is_valid_dns_server("192.0.2.1")
    assert is_valid_dns_server("2001:db8::53")
    assert not is_valid_dns_server("router.example.lan")
    assert not is_valid_dns_server("192.0.2.1:53")


def test_chain_prefers_mikrotik_over_dns():
    mikrotik = lambda mac, ip: "from-dhcp" if ip == "192.0.2.10" else None
    dns = lambda mac, ip: "from-dns"
    lookup = chain_hostname_lookups(mikrotik, dns)
    assert lookup("AA:BB:CC:DD:EE:01", "192.0.2.10") == "from-dhcp"
    assert lookup("AA:BB:CC:DD:EE:02", "192.0.2.11") == "from-dns"
    assert chain_hostname_lookups(None, dns)(None, "192.0.2.11") == "from-dns"


def test_fill_missing_hostnames_never_overwrites():
    clients = [
        {"mac": "AA:BB:CC:DD:EE:01", "ip": "192.0.2.10", "hostname": "from-dhcp"},
        {"mac": "AA:BB:CC:DD:EE:02", "ip": "192.0.2.11", "hostname": None},
        {"mac": "AA:BB:CC:DD:EE:03", "ip": "0.0.0.0", "hostname": None},
    ]
    lookup = lambda mac, ip: None if ip == "0.0.0.0" else "from-dns"
    assert fill_missing_hostnames(clients, lookup) is True
    assert [c["hostname"] for c in clients] == ["from-dhcp", "from-dns", None]
    assert fill_missing_hostnames(clients, lookup) is False
    assert fill_missing_hostnames(None, lookup) is False


# ---------------------------------------------------------------------------
# Resolver: cache, TTL, failures
# ---------------------------------------------------------------------------

class FakeHass:
    """Just enough of HomeAssistant for the resolver."""

    def __init__(self):
        self.tasks = []

    def async_create_background_task(self, coro, name):
        task = asyncio.get_running_loop().create_task(coro, name=name)
        self.tasks.append(task)
        return task

    async def drain(self):
        while self.tasks:
            await self.tasks.pop(0)


class FakeClock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now


def _make(answers, strip_domain=True):
    """Resolver whose lookups come from `answers` (ip -> name | Exception)."""
    hass = FakeHass()
    clock = FakeClock()
    calls = []

    async def query(ip):
        calls.append(ip)
        answer = answers.get(ip)
        if isinstance(answer, Exception):
            raise answer
        return answer

    resolver = DnsHostnameResolver(
        hass, server="192.0.2.1", strip_domain=strip_domain, query_func=query, time_func=clock
    )
    return resolver, hass, clock, calls


def test_resolver_caches_hits_and_respects_ttl():
    async def run():
        answers = {"192.0.2.10": "printer.example.lan."}
        resolver, hass, clock, calls = _make(answers)
        updates = []
        resolver.set_update_callback(lambda: updates.append(1))

        assert resolver.get_hostname(None, "192.0.2.10") is None
        resolver.async_request(["192.0.2.10", "192.0.2.10"])
        await hass.drain()
        assert calls == ["192.0.2.10"]
        assert resolver.get_hostname("AA:BB:CC:DD:EE:01", "192.0.2.10") == "printer"
        assert updates == [1]

        # Within the positive TTL: no new query.
        clock.now += DNS_POSITIVE_CACHE_TTL - 1
        resolver.async_request(["192.0.2.10"])
        await hass.drain()
        assert calls == ["192.0.2.10"]

        # Expired: re-queried, old name stays visible meanwhile.
        clock.now += 2
        answers["192.0.2.10"] = "printer2.example.lan"
        resolver.async_request(["192.0.2.10"])
        assert resolver.get_hostname(None, "192.0.2.10") == "printer"
        await hass.drain()
        assert calls == ["192.0.2.10", "192.0.2.10"]
        assert resolver.get_hostname(None, "192.0.2.10") == "printer2"

        diag = resolver.get_diagnostics()
        assert diag["hits"] == 2 and diag["errors"] == 0
        assert diag["cache_size"] == 1
        assert "192.0.2.10" not in repr(diag) and "printer" not in repr(diag)

    asyncio.run(run())


def test_resolver_keeps_full_name_when_not_stripping():
    async def run():
        resolver, hass, _clock, _calls = _make(
            {"192.0.2.10": "printer.example.lan."}, strip_domain=False
        )
        resolver.async_request(["192.0.2.10"])
        await hass.drain()
        assert resolver.get_hostname(None, "192.0.2.10") == "printer.example.lan"

    asyncio.run(run())


def test_resolver_skips_unspecified_and_empty_ips():
    async def run():
        resolver, hass, _clock, calls = _make({})
        resolver.async_request(["0.0.0.0", "", None, "Unknown"])
        await hass.drain()
        assert calls == []
        assert hass.tasks == []
        assert resolver.get_hostname(None, "0.0.0.0") is None

    asyncio.run(run())


def test_resolver_negative_cache_for_nxdomain():
    async def run():
        resolver, hass, clock, calls = _make({"192.0.2.11": None})
        updates = []
        resolver.set_update_callback(lambda: updates.append(1))
        resolver.async_request(["192.0.2.11"])
        await hass.drain()
        assert resolver.get_hostname(None, "192.0.2.11") is None
        assert updates == []

        clock.now += DNS_NEGATIVE_CACHE_TTL - 1
        resolver.async_request(["192.0.2.11"])
        await hass.drain()
        assert calls == ["192.0.2.11"]

        clock.now += 2
        resolver.async_request(["192.0.2.11"])
        await hass.drain()
        assert calls == ["192.0.2.11", "192.0.2.11"]
        assert resolver.get_diagnostics()["not_found"] == 2

    asyncio.run(run())


def test_resolver_timeout_does_not_crash_and_keeps_known_name():
    async def run():
        answers = {
            "192.0.2.10": "printer.example.lan",
            "192.0.2.12": DnsLookupError("timeout"),
        }
        resolver, hass, clock, calls = _make(answers)
        resolver.async_request(["192.0.2.10", "192.0.2.12"])
        await hass.drain()
        assert resolver.get_hostname(None, "192.0.2.10") == "printer"
        assert resolver.get_hostname(None, "192.0.2.12") is None

        # The server goes away: a known name survives the failed refresh.
        clock.now += DNS_POSITIVE_CACHE_TTL + 1
        answers["192.0.2.10"] = OSError("network unreachable")
        resolver.async_request(["192.0.2.10"])
        await hass.drain()
        assert resolver.get_hostname(None, "192.0.2.10") == "printer"

        diag = resolver.get_diagnostics()
        assert diag["errors"] == 2
        assert diag["last_error"] == "OSError"

        # A failure is retried only after the negative TTL.
        before = len(calls)
        resolver.async_request(["192.0.2.10"])
        await hass.drain()
        assert len(calls) == before
        clock.now += DNS_NEGATIVE_CACHE_TTL + 1
        resolver.async_request(["192.0.2.10"])
        await hass.drain()
        assert len(calls) == before + 1

    asyncio.run(run())


def test_resolver_limits_concurrency():
    async def run():
        from ha_zyxel.const import DNS_MAX_CONCURRENT_QUERIES

        active = 0
        peak = 0

        async def slow_query(ip):
            nonlocal active, peak
            active += 1
            peak = max(peak, active)
            await asyncio.sleep(0.01)
            active -= 1
            return None

        hass = FakeHass()
        resolver = DnsHostnameResolver(hass, query_func=slow_query, time_func=FakeClock())
        resolver.async_request(f"192.0.2.{i}" for i in range(1, 30))
        await hass.drain()
        assert peak == DNS_MAX_CONCURRENT_QUERIES

    asyncio.run(run())


def test_resolver_shutdown_cancels_pending_lookups():
    async def run():
        async def never(ip):
            await asyncio.sleep(3600)

        hass = FakeHass()
        resolver = DnsHostnameResolver(hass, query_func=never, time_func=FakeClock())
        resolver.async_request(["192.0.2.10"])
        await asyncio.sleep(0)
        await resolver.async_shutdown()
        assert resolver.get_hostname(None, "192.0.2.10") is None

    asyncio.run(run())


def test_api_uses_chained_lookup_mikrotik_first():
    """End to end through the Zyxel client parser: MikroTik name wins, DNS fills the rest."""
    from ha_zyxel.zyxel_ssh_api import ZyxelSSHAPI

    output = (
        "index: 0\n  MAC: aa:bb:cc:dd:ee:01\n  IPv4: 192.0.2.10\n  SSID: Home\n"
        "index: 1\n  MAC: aa:bb:cc:dd:ee:02\n  IPv4: 192.0.2.11\n  SSID: Home\n"
        "index: 2\n  MAC: aa:bb:cc:dd:ee:03\n  IPv4: 0.0.0.0\n  SSID: Home\n"
    )
    mikrotik = lambda mac, ip: "dhcp-name" if mac == "AA:BB:CC:DD:EE:01" else None
    dns_cache = {"192.0.2.10": "dns-name", "192.0.2.11": "printer"}
    dns = lambda mac, ip: dns_cache.get(ip)

    api = ZyxelSSHAPI("192.0.2.2", "admin", "dummy-password")
    api.set_hostname_resolver(chain_hostname_lookups(mikrotik, dns))
    clients = api._parse_clients(output)
    assert [c["hostname"] for c in clients] == ["dhcp-name", "printer", None]
