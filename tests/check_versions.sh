#!/bin/sh
set -eu

version=$(sed -n 's/^[[:space:]]*\.version = "\([^"]*\)".*/\1/p' build.zig.zon)
if [ -z "$version" ]; then
    echo "could not read project version from build.zig.zon" >&2
    exit 1
fi

check() {
    file=$1
    needle=$2
    if ! grep -Fq "$needle" "$file"; then
        echo "$file does not advertise core version $version" >&2
        exit 1
    fi
}

check README.md "v$version"
check bindings/python/pyproject.toml "version = \"$version\""
check bindings/nodejs/package.json "\"version\": \"$version\""
check bindings/nodejs/package-lock.json "\"version\": \"$version\""
check bindings/rust/Cargo.toml "version = \"$version\""
check bindings/ruby/lib/ztok/version.rb "VERSION = \"$version\""
check bindings/ruby/Gemfile.lock "ztok ($version)"
check bindings/java/pom.xml "<version>$version</version>"
check bindings/dotnet/Ztok/Ztok.csproj "<Version>$version</Version>"

echo "all release metadata matches $version"
