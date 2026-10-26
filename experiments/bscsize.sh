#!/bin/zsh
# usage: bscsize.sh "<bsc opts>" files...   -> prints "<compressed bytes>\t<file>" and a total
opts=$1; shift; tot=0; tmp=$(mktemp -d)
for f in "$@"; do
  bsc e "$f" "$tmp/o" ${=opts} >/dev/null 2>&1 || { echo "bsc failed on $f"; continue; }
  s=$(stat -f %z "$tmp/o"); tot=$((tot+s)); printf "%10d\t%s\n" $s "${f:t}"
done
printf "%10d\tTOTAL\n" $tot; rm -rf "$tmp"
