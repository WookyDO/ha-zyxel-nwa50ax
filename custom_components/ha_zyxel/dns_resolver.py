"""Optional client hostname resolver based on reverse DNS (PTR) lookups.

Alternative to the MikroTik resolver for networks where the router that hands
out DHCP leases also answers PTR queries for its LAN clients (e.g. a FRITZ!Box
answers "name.fritz.box" for every DHCP client). No SSH, no credentials.

Like the MikroTik resolver, this module is fully decoupled from the Zyxel data
collection:
- `get_hostname()` is a synchronous, network-free cache lookup that the Zyxel
  API calls while parsing the client list;
- lookups for IPs missing from the cache (or expired) are requested with
  `async_request()` and run as a background task, bounded by a semaphore and
  a per-query timeout. A slow or dead DNS server can never delay or fail an
  AP refresh: the worst case is "no hostname" for that client.

Two lookup back ends:
- no server configured: Home Assistant's system resolver, via
  `socket.gethostbyaddr` in the executor (it may block, so never on the loop);
- a server configured: a minimal PTR query sent over UDP straight to that
  server (standard library only, see `build_ptr_query` / `parse_ptr_response`).
"""
from __future__ import annotations

import asyncio
import ipaddress
import logging
import random
import socket
import struct
import time
from collections.abc import Awaitable, Callable, Iterable
from typing import Any, Optional

from homeassistant.core import HomeAssistant, callback

from .const import (
    DNS_CACHE_MAX_ENTRIES,
    DNS_MAX_CONCURRENT_QUERIES,
    DNS_NEGATIVE_CACHE_TTL,
    DNS_PORT,
    DNS_POSITIVE_CACHE_TTL,
    DNS_QUERY_TIMEOUT,
    DNS_SYSTEM_QUERY_TIMEOUT,
)

_LOGGER = logging.getLogger(__name__)

HostnameLookup = Callable[[Optional[str], Optional[str]], Optional[str]]

_DNS_TYPE_PTR = 12
_DNS_CLASS_IN = 1
_DNS_FLAG_QR = 0x8000
_DNS_FLAG_TC = 0x0200
_DNS_FLAG_RD = 0x0100
_DNS_RCODE_NXDOMAIN = 3
_MAX_NAME_LENGTH = 255
_MAX_POINTER_JUMPS = 32


class DnsLookupError(Exception):
    """A PTR lookup failed (timeout, server error, malformed answer).

    Distinct from "no such name": a failed lookup keeps a previously known
    hostname, a negative answer clears it.
    """


class DnsFormatError(DnsLookupError):
    """The DNS answer could not be parsed."""


class DnsServerError(DnsLookupError):
    """The DNS server answered with an error code other than NXDOMAIN."""


# ---------------------------------------------------------------------------
# Wire format (RFC 1035) - pure functions, unit tested with fixed bytes
# ---------------------------------------------------------------------------

def reverse_pointer_name(ip: str) -> str:
    """'192.0.2.10' -> '10.2.0.192.in-addr.arpa' (IPv6: ip6.arpa)."""
    return ipaddress.ip_address(ip).reverse_pointer


def _encode_name(name: str) -> bytes:
    encoded = b""
    for label in name.rstrip(".").split("."):
        raw = label.encode("ascii")
        if not raw or len(raw) > 63:
            raise ValueError(f"invalid DNS label in {name!r}")
        encoded += bytes([len(raw)]) + raw
    return encoded + b"\x00"


def build_ptr_query(qname: str, txid: int) -> bytes:
    """Standard recursive query: one question, QTYPE=PTR, QCLASS=IN."""
    header = struct.pack(">HHHHHH", txid & 0xFFFF, _DNS_FLAG_RD, 1, 0, 0, 0)
    return header + _encode_name(qname) + struct.pack(">HH", _DNS_TYPE_PTR, _DNS_CLASS_IN)


def _read_name(data: bytes, offset: int) -> tuple[str, int]:
    """Read a (possibly compressed) domain name.

    Returns the name and the offset right after it in the original position
    (i.e. after the first pointer when compression is used).
    """
    labels: list[str] = []
    length_total = 0
    jumps = 0
    end_offset: Optional[int] = None

    while True:
        if offset >= len(data):
            raise DnsFormatError("name runs past end of message")
        length = data[offset]
        if length & 0xC0 == 0xC0:
            if offset + 1 >= len(data):
                raise DnsFormatError("truncated compression pointer")
            jumps += 1
            if jumps > _MAX_POINTER_JUMPS:
                raise DnsFormatError("compression pointer loop")
            if end_offset is None:
                end_offset = offset + 2
            offset = ((length & 0x3F) << 8) | data[offset + 1]
            continue
        if length & 0xC0:
            raise DnsFormatError("unsupported label type")
        offset += 1
        if length == 0:
            break
        if offset + length > len(data):
            raise DnsFormatError("label runs past end of message")
        length_total += length + 1
        if length_total > _MAX_NAME_LENGTH:
            raise DnsFormatError("name too long")
        labels.append(data[offset:offset + length].decode("ascii", errors="replace"))
        offset += length

    return ".".join(labels), (end_offset if end_offset is not None else offset)


def parse_ptr_response(data: bytes, txid: int) -> Optional[str]:
    """Return the PTR target of a response, or None for NXDOMAIN / no PTR record.

    Raises DnsLookupError (or a subclass) for anything that is not a usable
    answer: wrong transaction id, not a response, truncated, server error,
    malformed message.
    """
    if len(data) < 12:
        raise DnsFormatError("message shorter than header")
    rid, flags, qdcount, ancount, _nscount, _arcount = struct.unpack(">HHHHHH", data[:12])
    if rid != (txid & 0xFFFF):
        raise DnsFormatError("transaction id mismatch")
    if not flags & _DNS_FLAG_QR:
        raise DnsFormatError("not a response")
    if flags & _DNS_FLAG_TC:
        raise DnsLookupError("truncated response")
    rcode = flags & 0x000F
    if rcode == _DNS_RCODE_NXDOMAIN:
        return None
    if rcode != 0:
        raise DnsServerError(f"server returned rcode {rcode}")

    offset = 12
    for _ in range(qdcount):
        _name, offset = _read_name(data, offset)
        offset += 4  # QTYPE + QCLASS
    for _ in range(ancount):
        _name, offset = _read_name(data, offset)
        if offset + 10 > len(data):
            raise DnsFormatError("truncated resource record")
        rtype, rclass, _ttl, rdlength = struct.unpack(">HHIH", data[offset:offset + 10])
        offset += 10
        if offset + rdlength > len(data):
            raise DnsFormatError("truncated resource data")
        if rtype == _DNS_TYPE_PTR and rclass == _DNS_CLASS_IN:
            target, _ = _read_name(data, offset)
            return target or None
        offset += rdlength  # e.g. a CNAME (RFC 2317 classless delegation)
    return None


class _PtrProtocol(asyncio.DatagramProtocol):
    """Collects the first datagram carrying the expected transaction id."""

    def __init__(self, txid: int, future: asyncio.Future) -> None:
        self._txid = txid
        self._future = future

    def datagram_received(self, data: bytes, addr: Any) -> None:
        if self._future.done() or len(data) < 2:
            return
        if struct.unpack(">H", data[:2])[0] != self._txid:
            return  # stray datagram, keep waiting until the timeout
        self._future.set_result(data)

    def error_received(self, exc: Exception) -> None:
        if not self._future.done():
            self._future.set_exception(DnsLookupError(f"socket error: {exc}"))

    def connection_lost(self, exc: Optional[Exception]) -> None:
        if not self._future.done():
            self._future.set_exception(DnsLookupError("socket closed"))


async def async_query_ptr(
    server: str, ip: str, timeout: float = DNS_QUERY_TIMEOUT, port: int = DNS_PORT
) -> Optional[str]:
    """Send one PTR query for `ip` to `server` over UDP and parse the answer."""
    loop = asyncio.get_running_loop()
    txid = random.getrandbits(16)
    query = build_ptr_query(reverse_pointer_name(ip), txid)
    future: asyncio.Future = loop.create_future()
    family = socket.AF_INET6 if ipaddress.ip_address(server).version == 6 else socket.AF_INET
    # A connected UDP socket only accepts datagrams from the server itself.
    transport, _protocol = await loop.create_datagram_endpoint(
        lambda: _PtrProtocol(txid, future), remote_addr=(server, port), family=family
    )
    try:
        transport.sendto(query)
        try:
            data = await asyncio.wait_for(future, timeout)
        except asyncio.TimeoutError as err:
            raise DnsLookupError("timeout") from err
    finally:
        transport.close()
    return parse_ptr_response(data, txid)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def normalize_hostname(name: Optional[str], strip_domain: bool) -> Optional[str]:
    """Clean a PTR answer for display.

    With `strip_domain`, keep only the first label ("printer.example.lan" ->
    "printer"). Answers that are just an IP literal are discarded.
    """
    if not name:
        return None
    name = name.strip().rstrip(".")
    if not name:
        return None
    try:
        ipaddress.ip_address(name)
        return None
    except ValueError:
        pass
    if strip_domain:
        name = name.split(".", 1)[0]
    return name or None


def resolvable_ip(ip: Optional[str]) -> Optional[str]:
    """The IP if it is worth a PTR query, else None (empty, 0.0.0.0, invalid)."""
    if not ip:
        return None
    try:
        addr = ipaddress.ip_address(ip.strip())
    except ValueError:
        return None
    if addr.is_unspecified or addr.is_multicast:
        return None
    return str(addr)


def chain_hostname_lookups(*lookups: Optional[HostnameLookup]) -> HostnameLookup:
    """First non-empty answer wins, in the given order (MikroTik before DNS)."""
    active = [lookup for lookup in lookups if lookup is not None]

    def _lookup(mac: Optional[str], ip: Optional[str]) -> Optional[str]:
        for lookup in active:
            name = lookup(mac, ip)
            if name:
                return name
        return None

    return _lookup


def fill_missing_hostnames(clients: Optional[list[dict[str, Any]]], lookup: HostnameLookup) -> bool:
    """Set `hostname` in place on clients that do not have one yet.

    Hostnames already present (e.g. from the MikroTik resolver) are never
    overwritten. Returns True if at least one client changed.
    """
    changed = False
    for client in clients or []:
        if client.get("hostname"):
            continue
        name = lookup(client.get("mac"), client.get("ip"))
        if name:
            client["hostname"] = name
            changed = True
    return changed


# ---------------------------------------------------------------------------
# Resolver
# ---------------------------------------------------------------------------

class DnsHostnameResolver:
    """Per-IP PTR cache, filled in the background."""

    def __init__(
        self,
        hass: HomeAssistant,
        server: Optional[str] = None,
        strip_domain: bool = True,
        query_func: Optional[Callable[[str], Awaitable[Optional[str]]]] = None,
        time_func: Callable[[], float] = time.monotonic,
    ) -> None:
        self.hass = hass
        self.server = (server or "").strip() or None
        self.strip_domain = strip_domain
        self._time = time_func
        self._query = query_func or (self._query_server if self.server else self._query_system)

        # ip -> (hostname or None, expiry timestamp)
        self._cache: dict[str, tuple[Optional[str], float]] = {}
        self._in_flight: set[str] = set()
        self._semaphore = asyncio.Semaphore(DNS_MAX_CONCURRENT_QUERIES)
        self._tasks: set[asyncio.Task] = set()
        self._on_update: Callable[[], None] = lambda: None

        self._hits = 0
        self._not_found = 0
        self._errors = 0
        self._last_error: Optional[str] = None

    def set_update_callback(self, on_update: Optional[Callable[[], None]]) -> None:
        """Called on the event loop after a batch produced new/changed names."""
        self._on_update = on_update or (lambda: None)

    # -- lookup back ends ---------------------------------------------------

    async def _query_server(self, ip: str) -> Optional[str]:
        return await async_query_ptr(self.server, ip)

    async def _query_system(self, ip: str) -> Optional[str]:
        def _lookup() -> Optional[str]:
            try:
                return socket.gethostbyaddr(ip)[0]
            except socket.herror as err:
                if err.errno == 1:  # HOST_NOT_FOUND: authoritative "no name"
                    return None
                raise DnsLookupError(f"resolver error {err.errno}") from err
            except OSError as err:
                raise DnsLookupError(str(err)) from err

        try:
            return await asyncio.wait_for(
                self.hass.async_add_executor_job(_lookup), DNS_SYSTEM_QUERY_TIMEOUT
            )
        except asyncio.TimeoutError as err:
            raise DnsLookupError("timeout") from err

    # -- public API ---------------------------------------------------------

    def get_hostname(self, mac: Optional[str], ip: Optional[str]) -> Optional[str]:
        """Synchronous cache lookup, no I/O. `mac` is unused (PTR is per IP).

        An expired positive entry is still returned until its refresh has
        completed, so names do not blink while a re-lookup is in flight.
        """
        key = resolvable_ip(ip)
        if key is None:
            return None
        entry = self._cache.get(key)
        return entry[0] if entry else None

    @callback
    def async_request(self, ips: Iterable[Optional[str]]) -> None:
        """Schedule background lookups for IPs that are missing or expired."""
        now = self._time()
        due: list[str] = []
        for raw in ips:
            ip = resolvable_ip(raw)
            if ip is None or ip in self._in_flight or ip in due:
                continue
            entry = self._cache.get(ip)
            if entry is not None and entry[1] > now:
                continue
            due.append(ip)
        if not due:
            return
        self._prune(now)
        self._in_flight.update(due)
        task = self.hass.async_create_background_task(
            self._async_resolve_batch(due), name="ha_zyxel dns hostname lookup"
        )
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def async_shutdown(self) -> None:
        for task in list(self._tasks):
            task.cancel()
        for task in list(self._tasks):
            try:
                await task
            except asyncio.CancelledError:
                pass
        self._tasks.clear()
        self._in_flight.clear()

    def get_diagnostics(self) -> dict[str, Any]:
        """Counters only - never hostnames or IPs."""
        return {
            "custom_server": self.server is not None,
            "strip_domain": self.strip_domain,
            "cache_size": len(self._cache),
            "cached_names": sum(1 for name, _ in self._cache.values() if name),
            "hits": self._hits,
            "not_found": self._not_found,
            "errors": self._errors,
            "last_error": self._last_error,
        }

    # -- internals ----------------------------------------------------------

    async def _async_resolve_batch(self, ips: list[str]) -> None:
        try:
            results = await asyncio.gather(*(self._async_resolve_one(ip) for ip in ips))
        finally:
            self._in_flight.difference_update(ips)
        if any(results):
            try:
                self._on_update()
            except Exception as err:  # never let a consumer break the resolver
                _LOGGER.debug("DNS resolver update callback failed: %s", err)

    async def _async_resolve_one(self, ip: str) -> bool:
        """Resolve one IP and update the cache. Returns True if the name changed."""
        previous = self._cache.get(ip, (None, 0.0))[0]
        async with self._semaphore:
            try:
                raw = await self._query(ip)
            except asyncio.CancelledError:
                raise
            except Exception as err:  # DnsLookupError, OSError, anything else
                self._errors += 1
                self._last_error = type(err).__name__
                _LOGGER.debug("PTR lookup failed (%s)", type(err).__name__)
                # Keep a name we already had; retry after the negative TTL.
                self._cache[ip] = (previous, self._time() + DNS_NEGATIVE_CACHE_TTL)
                return False

        name = normalize_hostname(raw, self.strip_domain)
        if name:
            self._hits += 1
            self._cache[ip] = (name, self._time() + DNS_POSITIVE_CACHE_TTL)
        else:
            self._not_found += 1
            self._cache[ip] = (None, self._time() + DNS_NEGATIVE_CACHE_TTL)
        return name != previous

    def _prune(self, now: float) -> None:
        """Bound memory: drop long-expired entries once the cache gets large."""
        if len(self._cache) < DNS_CACHE_MAX_ENTRIES:
            return
        cutoff = now - DNS_POSITIVE_CACHE_TTL
        for ip in [ip for ip, (_, exp) in self._cache.items() if exp < cutoff]:
            del self._cache[ip]
        if len(self._cache) >= DNS_CACHE_MAX_ENTRIES:
            for ip in sorted(self._cache, key=lambda k: self._cache[k][1])[
                : len(self._cache) - DNS_CACHE_MAX_ENTRIES + 1
            ]:
                del self._cache[ip]
