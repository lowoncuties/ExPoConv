#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
repository_root=$(cd -- "$script_dir/.." && pwd -P)
data_root=${POINCARE_EXEC_DATA_ROOT:-$repository_root/data/raw}
download_dir="$data_root/downloads"
isic_dir="$data_root/ISIC"
cifar_dir="$data_root/cifar-100-python"
cifar_archive="$download_dir/cifar-100-python.tar.gz"
cifar_extract_dir=""

mkdir -p "$download_dir" "$isic_dir"

cleanup() {
  if [[ -n "$cifar_extract_dir" && -d "$cifar_extract_dir" ]]; then
    rm -rf -- "$cifar_extract_dir"
  fi
}
trap cleanup EXIT

download() {
  local url=$1
  local destination=$2
  if [[ -f "$destination" ]]; then
    curl --fail --location --retry 5 --continue-at - --output "$destination" "$url"
  else
    curl --fail --location --retry 5 --output "$destination" "$url"
  fi
}

cifar_url=https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz
download "$cifar_url" "$cifar_archive"
printf '%s  %s\n' \
  '85cd44d02ba6437773c5bbd22e183051d648de2e7d6b014e1ef29b855ba677a7' \
  "$cifar_archive" | sha256sum --check

if [[ -d "$cifar_dir" ]]; then
  for required_file in train test meta; do
    if [[ ! -f "$cifar_dir/$required_file" ]]; then
      echo "Incomplete existing CIFAR-100 directory: $cifar_dir" >&2
      exit 1
    fi
  done
else
  cifar_extract_dir=$(mktemp -d "$data_root/.cifar-100-python.XXXXXX")
  tar \
    --extract \
    --gzip \
    --file "$cifar_archive" \
    --directory "$cifar_extract_dir" \
    --no-same-owner \
    --no-same-permissions
  for required_file in train test meta; do
    if [[ ! -f "$cifar_extract_dir/cifar-100-python/$required_file" ]]; then
      echo "CIFAR-100 archive is missing $required_file" >&2
      exit 1
    fi
  done
  mv "$cifar_extract_dir/cifar-100-python" "$cifar_dir"
  rmdir "$cifar_extract_dir"
  cifar_extract_dir=""
fi
for required_file in train test meta; do
  tar \
    --extract \
    --to-stdout \
    --gzip \
    --file "$cifar_archive" \
    "cifar-100-python/$required_file" \
    | cmp --silent - "$cifar_dir/$required_file"
done

isic_base=https://isic-archive.s3.amazonaws.com/challenges/2016
declare -Ar isic_sha256=(
  [ISBI2016_ISIC_Part1_Training_Data.zip]=ad5b48fb7d9073f901d7ea353cc965360c003eabbce1c1f9cdd1c3c42bd90ba2
  [ISBI2016_ISIC_Part1_Training_GroundTruth.zip]=81cff572b1cc0e6198ddddc1c8e14ea13dc9b10bfa316ee1877a4b183ddd876a
  [ISBI2016_ISIC_Part1_Test_Data.zip]=5cb5766aecc0c2981b6fae544de790908c9e2465186e94965bfe88b413f63c99
  [ISBI2016_ISIC_Part1_Test_GroundTruth.zip]=065df24f07424aea8fdfe9f392d70d1ff8fba21a66736e1d51dc8b108492020e
)
for archive in \
  ISBI2016_ISIC_Part1_Training_Data.zip \
  ISBI2016_ISIC_Part1_Training_GroundTruth.zip \
  ISBI2016_ISIC_Part1_Test_Data.zip \
  ISBI2016_ISIC_Part1_Test_GroundTruth.zip
do
  download "$isic_base/$archive" "$download_dir/$archive"
  printf '%s  %s\n' \
    "${isic_sha256[$archive]}" \
    "$download_dir/$archive" | sha256sum --check
  unzip -n -q "$download_dir/$archive" -d "$isic_dir"
done

(
  cd "$data_root"
  sha256sum \
    downloads/cifar-100-python.tar.gz \
    downloads/ISBI2016_ISIC_Part1_*.zip
) > "$data_root/dataset_sources.sha256"
echo "ISIC16 and CIFAR-100 datasets available below $data_root"
