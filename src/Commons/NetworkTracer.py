from __future__ import annotations

import ipaddress
import socket
import threading
import time
import traceback
from datetime import datetime
from typing import Any, Callable, ClassVar, Optional, cast

# Typeshed exposes socket._Address on many versions, but not all.
try:
    from socket import _Address as Address  # type: ignore[attr-defined]
except Exception:
    Address = Any  # fallback

ConnectFn = Callable[[socket.socket, Address], Any]
ConnectExFn = Callable[[socket.socket, Address], int]
GetAddrInfoFn = Callable[..., Any]
BindFn = Callable[[socket.socket, Address], Any]
ListenFn = Callable[[socket.socket, int], None]

# Pylance-friendly aliases for getaddrinfo results.
# We keep sockaddr as "object" and narrow at runtime to avoid Unknown.
SockAddr = object
AddrInfo = tuple[socket.AddressFamily, socket.SocketKind, int, str, SockAddr]
AddrInfoList = list[AddrInfo]


class NetworkTracer:
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"
    RED = "\033[31m"
    RESET = "\033[0m"
    _original_connect: ClassVar[ConnectFn] = cast(ConnectFn, socket.socket.connect)
    _original_connect_ex: ClassVar[ConnectExFn] = cast(
        ConnectExFn, socket.socket.connect_ex
    )
    _original_getaddrinfo: ClassVar[GetAddrInfoFn] = cast(
        GetAddrInfoFn, socket.getaddrinfo
    )
    _original_bind: ClassVar[BindFn] = cast(BindFn, socket.socket.bind)
    _original_listen: ClassVar[ListenFn] = socket.socket.listen

    _patched: ClassVar[bool] = False
    _lock: ClassVar[threading.Lock] = threading.Lock()

    @staticmethod
    def _now_iso() -> str:
        return datetime.fromtimestamp(time.time()).isoformat()

    @staticmethod
    def _is_ip(value: str) -> bool:
        try:
            ipaddress.ip_address(value)
            return True
        except ValueError:
            return False

    @staticmethod
    def _extract_host_port(address: Address) -> tuple[Optional[str], Optional[int]]:
        """
        Extract host/port from socket 'address' which can be:
          - (host, port) for AF_INET
          - (host, port, flowinfo, scopeid) for AF_INET6
          - str path for AF_UNIX
        """
        if isinstance(address, str):
            return address, None

        host: Optional[str] = None
        port: Optional[int] = None

        # Pylance strict: after isinstance(..., tuple), element types are Unknown.
        # Cast to tuple[object, ...] before len()/indexing.
        if isinstance(address, tuple):
            a = cast(tuple[object, ...], address)
            if len(a) >= 2:
                h, p = a[0], a[1]
                if isinstance(h, str):
                    host = h
                if isinstance(p, int):
                    port = p

        return host, port

    @staticmethod
    def _print_stack() -> None:
        for frame in traceback.extract_stack():
            filename = frame.filename
            normalized = filename.replace("\\", "/")
            if (
                "site-packages" in normalized
                or "/lib/python" in normalized
                or filename.startswith("<frozen ")
            ):
                continue
            print(f" {filename}:{frame.lineno} in {frame.name}")

    @staticmethod
    def _traced_getaddrinfo(
        host: object,
        port: object,
        family: int = 0,
        type: int = 0,
        proto: int = 0,
        flags: int = 0,
    ) -> Any:
        """
        Trace forward DNS / address resolution (hostname -> IP).
        """
        print("\n=== DNS resolution detected (getaddrinfo) ===")
        print(f"Time: {NetworkTracer._now_iso()}")
        print(
            f"Query: {NetworkTracer.MAGENTA}host={host!r}, port={port!r}, family={family}, type={type}, proto={proto}, flags={flags}{NetworkTracer.RESET}"
        )

        try:
            result = NetworkTracer._original_getaddrinfo(
                host, port, family, type, proto, flags
            )

            infos = cast(AddrInfoList, result)
            ips: set[str] = set()

            for _fam, _kind, _proto, _canon, sockaddr in infos:
                # sockaddr can be various tuple shapes; narrow safely.
                if isinstance(sockaddr, tuple):
                    sa = cast(
                        tuple[object, ...], sockaddr
                    )  # <-- removes tuple[Unknown, ...]
                    if len(sa) >= 1:
                        addr0 = sa[0]
                        if isinstance(addr0, str):
                            ips.add(addr0)

            if isinstance(host, str) and ips:
                print(
                    f"Resolved {NetworkTracer.MAGENTA}{host!r} -> {sorted(ips)}{NetworkTracer.RESET}"
                )
            elif ips:
                print(
                    f"Resolved -> {NetworkTracer.MAGENTA}{sorted(ips)}{NetworkTracer.RESET}"
                )
            else:
                print(
                    f"Resolved -> {NetworkTracer.MAGENTA}(no addresses extracted){NetworkTracer.RESET}"
                )

            NetworkTracer._print_stack()
            print("===========================================\n")
            return result

        except Exception as e:
            print(f"getaddrinfo failed: {e}")
            NetworkTracer._print_stack()
            print("===========================================\n")
            raise

    @staticmethod
    def _traced_connect(self_sock: socket.socket, address: Address) -> Any:
        """
        Trace actual connection attempts. Often receives an IP (DNS already done),
        so we do reverse DNS for IPs, forward DNS for hostnames.
        """
        print("\n=== Network connection attempt detected (connect) ===")
        print(f"Time: {NetworkTracer._now_iso()}")
        print(f"Destination: {NetworkTracer.CYAN}{address!r}{NetworkTracer.RESET}")

        host, port = NetworkTracer._extract_host_port(address)

        if host is not None:
            try:
                if NetworkTracer._is_ip(host):
                    # Reverse DNS: IP -> name
                    name, _, _ = socket.gethostbyaddr(host)
                    suffix = f":{port}" if port is not None else ""
                    print(
                        f"Reverse DNS: {NetworkTracer.MAGENTA}{host}{suffix} -> {name}{NetworkTracer.RESET}"
                    )
                else:
                    # Forward DNS: name -> addresses (IPv4 + IPv6)
                    infos_any = socket.getaddrinfo(host, port)
                    infos = cast(AddrInfoList, infos_any)

                    addrs: set[str] = set()
                    for info in infos:
                        sockaddr = info[4]
                        if isinstance(sockaddr, tuple):
                            sa = cast(tuple[object, ...], sockaddr)
                            if len(sa) >= 1:
                                addr0 = sa[0]
                                if isinstance(addr0, str):
                                    addrs.add(addr0)

                    suffix = f":{port}" if port is not None else ""
                    print(
                        f"Forward DNS: {NetworkTracer.MAGENTA}{host}{suffix} -> {sorted(addrs)}{NetworkTracer.RESET}"
                    )

            except Exception as e:
                print(f"DNS lookup failed for {host!r}: {e}")

        NetworkTracer._print_stack()
        print("===============================================\n")

        return NetworkTracer._original_connect(self_sock, address)

    # Loopback addresses used for harmless IPv6/IPv4 capability probes.
    _LOOPBACK_ADDRS: ClassVar[frozenset[str]] = frozenset({"127.0.0.1", "::1"})

    @staticmethod
    def _traced_bind(self_sock: socket.socket, address: Address) -> Any:
        """
        Trace socket.bind() calls.  A 3rd-party component binding to a
        local address is a potential indicator of a backdoor listener.
        """
        host, port = NetworkTracer._extract_host_port(address)

        # Annotate known-safe loopback probes (e.g. urllib3 IPv6 detection).
        label = ""
        if host in NetworkTracer._LOOPBACK_ADDRS and port == 0:
            label = (
                f"  {NetworkTracer.RESET}[loopback + ephemeral port — "
                f"typical IPv6 capability probe, likely harmless]"
            )

        print(f"\n=== {NetworkTracer.RED}Socket BIND detected{NetworkTracer.RESET} ===")
        print(f"Time: {NetworkTracer._now_iso()}")
        print(
            f"Bind address: {NetworkTracer.RED}{address!r}{label}{NetworkTracer.RESET}"
        )

        if host is not None:
            suffix = f":{port}" if port is not None else ""
            print(
                f"Local endpoint: {NetworkTracer.RED}{host}{suffix}{NetworkTracer.RESET}"
            )

        NetworkTracer._print_stack()
        print("===============================================\n")

        return NetworkTracer._original_bind(self_sock, address)

    @staticmethod
    def _traced_listen(self_sock: socket.socket, backlog: int = 1) -> None:
        """
        Trace socket.listen() calls.  If any code starts listening for
        inbound connections this will be reported immediately.
        """
        # Retrieve the local address the socket is bound to (if available).
        try:
            local_addr = self_sock.getsockname()
        except Exception:
            local_addr = "(unknown)"

        print(
            f"\n=== {NetworkTracer.RED}Socket LISTEN detected — "
            f"port open for inbound connections{NetworkTracer.RESET} ==="
        )
        print(f"Time: {NetworkTracer._now_iso()}")
        print(f"Local address: {NetworkTracer.RED}{local_addr!r}{NetworkTracer.RESET}")
        print(f"Backlog: {NetworkTracer.RED}{backlog}{NetworkTracer.RESET}")

        NetworkTracer._print_stack()
        print("===============================================\n")

        return NetworkTracer._original_listen(self_sock, backlog)

    @staticmethod
    def _traced_connect_ex(self_sock: socket.socket, address: Address) -> int:
        """
        Trace non-blocking connection attempts via connect_ex().
        connect_ex returns 0 on success or an errno on failure instead of raising.
        """
        print("\n=== Network connection attempt detected (connect_ex) ===")
        print(f"Time: {NetworkTracer._now_iso()}")
        print(f"Destination: {NetworkTracer.CYAN}{address!r}{NetworkTracer.RESET}")

        host, port = NetworkTracer._extract_host_port(address)

        if host is not None:
            try:
                if NetworkTracer._is_ip(host):
                    # Reverse DNS: IP -> name
                    name, _, _ = socket.gethostbyaddr(host)
                    suffix = f":{port}" if port is not None else ""
                    print(
                        f"Reverse DNS: {NetworkTracer.CYAN}{host}{suffix} -> {name}{NetworkTracer.RESET}"
                    )
                else:
                    # Forward DNS: name -> addresses (IPv4 + IPv6)
                    infos_any = socket.getaddrinfo(host, port)
                    infos = cast(AddrInfoList, infos_any)

                    addrs: set[str] = set()
                    for info in infos:
                        sockaddr = info[4]
                        if isinstance(sockaddr, tuple):
                            sa = cast(tuple[object, ...], sockaddr)
                            if len(sa) >= 1:
                                addr0 = sa[0]
                                if isinstance(addr0, str):
                                    addrs.add(addr0)

                    suffix = f":{port}" if port is not None else ""
                    print(
                        f"Forward DNS: {NetworkTracer.CYAN}{host}{suffix} -> {sorted(addrs)}{NetworkTracer.RESET}"
                    )

            except Exception as e:
                print(f"DNS lookup failed for {host!r}: {e}")

        NetworkTracer._print_stack()
        print("===============================================\n")

        return NetworkTracer._original_connect_ex(self_sock, address)

    @classmethod
    def enable_tracer(cls) -> None:
        with cls._lock:
            if cls._patched:
                return

            # Refresh originals (in case something patched before us)
            cls._original_connect = cast(ConnectFn, socket.socket.connect)
            cls._original_connect_ex = cast(ConnectExFn, socket.socket.connect_ex)
            cls._original_getaddrinfo = cast(GetAddrInfoFn, socket.getaddrinfo)
            cls._original_bind = cast(BindFn, socket.socket.bind)
            cls._original_listen = socket.socket.listen

            # Monkeypatch (casts are intentional at the unsafe boundary)
            socket.socket.connect = cast(Any, cls._traced_connect)
            socket.socket.connect_ex = cast(Any, cls._traced_connect_ex)
            socket.getaddrinfo = cast(Any, cls._traced_getaddrinfo)
            socket.socket.bind = cast(Any, cls._traced_bind)
            socket.socket.listen = cast(Any, cls._traced_listen)

            cls._patched = True

    @classmethod
    def disable_tracer(cls) -> None:
        with cls._lock:
            if not cls._patched:
                return

            socket.socket.connect = cast(Any, cls._original_connect)
            socket.socket.connect_ex = cast(Any, cls._original_connect_ex)
            socket.getaddrinfo = cast(Any, cls._original_getaddrinfo)
            socket.socket.bind = cast(Any, cls._original_bind)
            socket.socket.listen = cast(Any, cls._original_listen)

            cls._patched = False

    @classmethod
    def is_tracer_enabled(cls) -> bool:
        return cls._patched
