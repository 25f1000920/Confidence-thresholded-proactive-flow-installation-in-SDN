import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from shared.live_utils import ip_to_host, host_to_ip, host_to_port, build_payload, parse_payload


def test_name_mapping_roundtrip():
    for i in range(1, 17):
        assert ip_to_host(f"10.0.0.{i}") == f"h{i}"
        assert host_to_ip(f"h{i}") == f"10.0.0.{i}"
        assert host_to_port(f"h{i}") == i
    print("test_name_mapping_roundtrip: PASS")


def test_payload_roundtrip_and_padding():
    p = build_payload(42, 1234.567891)
    assert parse_payload(p) == (42, 1234.567891)
    assert parse_payload(p + b"\x00" * 20) == (42, 1234.567891)  # trailing frame padding tolerated
    assert parse_payload(b"garbage") is None
    assert parse_payload(b"") is None
    print("test_payload_roundtrip_and_padding: PASS")


if __name__ == "__main__":
    test_name_mapping_roundtrip()
    test_payload_roundtrip_and_padding()
    print("\nAll live_utils tests passed.")
