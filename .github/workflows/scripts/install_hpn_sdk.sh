#!/bin/bash
# Install an explicitly selected HPN SDK, or retain the image-builtin package.
set -euo pipefail

url=${HPN_SDK_URL:-}
if [[ -z $url ]]; then
    echo 'HPN SDK source: image-builtin'
    exit 0
fi

python3 - "$url" <<'PY'
import sys
from urllib.parse import urlsplit

url = sys.argv[1]
try:
    parsed = urlsplit(url)
    valid = (parsed.scheme == 'https'
             and parsed.hostname == 'zone-ci.oss-cn-hangzhou.aliyuncs.com'
             and parsed.port in (None, 443)
             and parsed.username is None and parsed.password is None
             and parsed.path.endswith('.bin'))
except ValueError:
    valid = False
if not valid:
    raise SystemExit('HPN_SDK_URL must be an HTTPS .bin URL on zone-ci.oss-cn-hangzhou.aliyuncs.com')
PY

installer=$(mktemp /tmp/hpnsdk.XXXXXX)
trap 'rm -f "$installer"' EXIT
printf 'HPN SDK URL: %s\n' "$url"
wget --tries=3 --timeout=60 --waitretry=5 -O "$installer" "$url"
bash "$installer" --install --force
echo 'HPN SDK installation completed'
