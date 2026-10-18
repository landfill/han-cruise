#!/bin/sh
node tools/capture.mjs --scenes ${1:-golden_city} --out captures/dev 2>&1 | grep -E "error [0-9]+:|^\[log\] \[|did not|ready|load failed|pageerror|GPU error" | grep -v "^\[warning\]" | head -${2:-30}
