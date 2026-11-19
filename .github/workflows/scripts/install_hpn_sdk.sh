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
from urllib.parse import urlsplit, urlunsplit

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
print('HPN SDK URL: ' + urlunsplit((parsed.scheme, parsed.netloc, parsed.path, '', '')))
PY

installer=$(mktemp /tmp/hpnsdk.XXXXXX)
trap 'rm -f "$installer"' EXIT
wget --quiet --max-redirect=0 --tries=3 --timeout=60 --waitretry=5 -O "$installer" "$url"
test -s "$installer" || { echo 'HPN SDK installer is empty' >&2; exit 1; }
bash "$installer" --install --force
echo 'HPN SDK installation completed'
