#!/usr/bin/env python3
"""
GPU Sparse IDS — Snort / Suricata Rule Importer
Parses standard Snort and Suricata rules (.rules files) and converts them
into the pipe-delimited signature format required by the GPU Sparse IDS:
  ID|NAME|PROTOCOL|PORT|HEX_PATTERN|DESCRIPTION
"""

import sys
import re
import argparse
import binascii

def parse_content(content_str):
    """
    Parses a Snort content string which may contain mixed text and hex bytes:
    e.g. "GET " or "|90 90 90 90|" or "USER |0d 0a|"
    Returns the concatenated hex string (lowercase).
    """
    # Remove outer quotes if present
    content_str = content_str.strip('"')
    hex_parts = []
    
    # Snort content can alternate between plain text and |hex bytes|
    tokens = re.split(r'(\|[^|]*\|)', content_str)
    for token in tokens:
        if not token:
            continue
        if token.startswith('|') and token.endswith('|'):
            # Hex token, e.g. "|90 90 90|"
            raw_hex = token[1:-1].replace(' ', '').replace('\t', '')
            hex_parts.append(raw_hex.lower())
        else:
            # Plain text token, e.g. "USER "
            raw_hex = binascii.hexlify(token.encode('latin1', errors='replace')).decode('ascii')
            hex_parts.append(raw_hex.lower())
            
    return "".join(hex_parts)

def parse_rule(line):
    """
    Parses a single Snort/Suricata rule line.
    Returns a dict or None if invalid/comment.
    """
    line = line.strip()
    if not line or line.startswith('#'):
        return None

    # Match: action proto src sport dir dst dport (options)
    match = re.match(r'^(alert|log|pass|drop)\s+(\w+)\s+(\S+)\s+(\S+)\s+(?:->|<>)\s+(\S+)\s+(\S+)\s*\((.*)\)$', line)
    if not match:
        return None

    action, proto_str, src_ip, src_port, dst_ip, dst_port_str, options_str = match.groups()

    # Protocol mapping: 0=any, 6=TCP, 17=UDP
    proto_str = proto_str.lower()
    if proto_str == 'tcp':
        protocol = 6
    elif proto_str == 'udp':
        protocol = 17
    else:
        protocol = 0

    # Destination port mapping
    try:
        dst_port = int(dst_port_str)
    except ValueError:
        # e.g., 'any', '$HTTP_PORTS', etc.
        dst_port = 0

    # Parse options in body: key:value; or bare flags;
    opts = {}
    contents = []
    
    # Regex to extract key[:value];
    # Handles escaped semicolons \; inside quotes
    opt_pattern = re.compile(r'([a-zA-Z0-9_\-]+)(?::\s*"((?:\\.|[^"\\])*)"|:\s*([^;]+))?\s*;')
    for m in opt_pattern.finditer(options_str):
        key = m.group(1).lower()
        val = m.group(2) if m.group(2) is not None else m.group(3)
        if val is not None:
            val = val.strip()
            
        if key == 'content':
            if val is not None:
                contents.append(val)
        else:
            opts[key] = val if val is not None else True

    # We need at least one content match to form an n-gram signature
    if not contents:
        return None

    # Use first content (or longest content) as primary pattern
    primary_content = max(contents, key=lambda c: len(parse_content(c)))
    hex_pattern = parse_content(primary_content)

    # Must have at least 2 bytes (4 hex chars)
    if len(hex_pattern) < 4:
        return None

    sid = opts.get('sid', '0')
    try:
        sid_num = int(sid)
    except ValueError:
        sid_num = 0

    msg = opts.get('msg', f'Rule_{sid}')
    if msg:
        msg = msg.strip('"')

    name = re.sub(r'[^a-zA-Z0-9_]', '_', msg)[:32].strip('_')
    if not name:
        name = f"SIG_{sid_num}"

    description = msg

    return {
        'id': sid_num,
        'name': name,
        'protocol': protocol,
        'dst_port': dst_port,
        'hex_pattern': hex_pattern,
        'description': description
    }

def convert_rules_file(input_path, output_path, min_pattern_bytes=3, max_rules=None):
    """
    Reads Snort/Suricata rules from input_path and writes GPU Sparse IDS signatures to output_path.
    """
    converted = []
    seen_ids = set()

    with open(input_path, 'r', encoding='utf-8', errors='ignore') as infile:
        for line in infile:
            rule = parse_rule(line)
            if not rule:
                continue

            # Minimum pattern length check
            if len(rule['hex_pattern']) < min_pattern_bytes * 2:
                continue

            # Ensure unique IDs
            if rule['id'] in seen_ids or rule['id'] == 0:
                rule['id'] = 200000 + len(converted) + 1
            seen_ids.add(rule['id'])

            converted.append(rule)
            if max_rules and len(converted) >= max_rules:
                break

    with open(output_path, 'w', encoding='utf-8') as outfile:
        outfile.write("# GPU Sparse IDS Signature Database (Imported from Snort/Suricata)\n")
        outfile.write("# Format: ID|NAME|PROTOCOL|PORT|HEX_PATTERN|DESCRIPTION\n")
        outfile.write("# PROTOCOL: 0=any, 6=TCP, 17=UDP  PORT: 0=any\n\n")
        for r in converted:
            outfile.write(f"{r['id']}|{r['name']}|{r['protocol']}|{r['dst_port']}|{r['hex_pattern']}|{r['description']}\n")

    return len(converted)

def generate_sample_rules_file(path):
    """
    Creates a sample Snort rule file for demonstration.
    """
    sample_rules = """# Sample Snort Rules
alert tcp $EXTERNAL_NET any -> $HTTP_SERVERS 80 (msg:"WEB-ATTACKS /etc/passwd access attempt"; flow:to_server,established; content:"/etc/passwd"; nocase; sid:100001; rev:1;)
alert tcp $EXTERNAL_NET any -> $HTTP_SERVERS 80 (msg:"WEB-ATTACKS union select sql injection"; flow:to_server,established; content:"union select"; nocase; sid:100002; rev:1;)
alert tcp $EXTERNAL_NET any -> $HTTP_SERVERS 8080 (msg:"MALWARE Shellcode NOP Sled"; content:"|90 90 90 90 90 90|"; sid:100003; rev:1;)
alert udp $EXTERNAL_NET any -> $HOME_NET 53 (msg:"PROTOCOL-DNS excessive TXT query payload"; content:"|00 00 10 00 01|"; sid:100004; rev:1;)
alert tcp $EXTERNAL_NET any -> $HTTP_SERVERS any (msg:"WEB-CLIENT script alert xss"; content:"<script>alert("; nocase; sid:100005; rev:1;)
"""
    with open(path, 'w', encoding='utf-8') as f:
        f.write(sample_rules)

def main():
    parser = argparse.ArgumentParser(description="Convert Snort / Suricata rules to GPU Sparse IDS format.")
    parser.add_argument("-i", "--input", help="Path to input .rules file")
    parser.add_argument("-o", "--output", help="Path to output signature file")
    parser.add_argument("--min-len", type=int, default=3, help="Minimum pattern byte length (default: 3)")
    parser.add_argument("--max-rules", type=int, default=None, help="Maximum rules to convert")
    parser.add_argument("--sample", action="store_true", help="Generate a sample .rules file and test conversion")

    args = parser.parse_args()

    if args.sample or not args.input:
        sample_in = "data/signatures/sample_snort.rules"
        sample_out = "data/signatures/imported_snort_signatures.txt"
        print(f"[*] Generating sample rules: {sample_in}")
        generate_sample_rules_file(sample_in)
        count = convert_rules_file(sample_in, sample_out, args.min_len, args.max_rules)
        print(f"[+] Successfully converted {count} Snort rules -> {sample_out}")
        print("\n--- Output Preview ---")
        with open(sample_out, 'r') as f:
            print(f.read().strip())
        return

    out_path = args.output or (args.input.rsplit('.', 1)[0] + "_sparse_ids.txt")
    count = convert_rules_file(args.input, out_path, args.min_len, args.max_rules)
    print(f"[+] Successfully converted {count} rules from {args.input} -> {out_path}")

if __name__ == "__main__":
    main()

