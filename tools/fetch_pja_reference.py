"""Fetch pinned, blob-verified PJA source for the development-only C oracle.

Not part of the Rust build or runtime. Upstream source retains its MIT
attribution; see crates/rs-jxl-plus/LICENSE-PJA and NOTICE-PJA.
Existing modified files are rejected and are never overwritten.
"""
from hashlib import sha1
from pathlib import Path
from urllib.request import Request, urlopen
import sys

REVISION = "1058f63fbfb9c60cc6640b7bb130d6f7c7ea7dbd"
BLOBS = {
    "b3chunks.c": "34e1e496ad74d7f1007f4b2102198cbe049a4f39",
    "pja.inc": "343687c3f230f10a11b5050642bcc6d01e7de899",
    "pjaarith.pas": "3cfc03038409910e5d904fac0f9b17ad8de3d62e",
    "pjabench.pas": "b6e14817d6dd0ceed3d3e8142f830c1658ab3709",
    "pjablake3.pas": "ad1754670d27c9bd05c855e9532c4cc03b680cf5",
    "pjacli.pas": "49a756699a66ba86c41013bd6ddd100a0eee9d4f",
    "pjacodec.pas": "3580a05e8e2f819ed4b04608c24dcee851443308",
    "pjacoef.inc": "783e55e8370458efe1549e76b4e4d86f6cc3ba9b",
    "pjaconsole.pas": "316919ffefd3375deedc6fda673e84f8465d1821",
    "pjacpu.pas": "617de85eaf78a759511431af41cebe823fcbeb9f",
    "pjaformat.pas": "69c361117911b8810eeb30fe3c29a35e9b47ee72",
    "pjafs.pas": "29844db150365b8098c57fafda751f36c6bc6401",
    "pjaheap.pas": "24cd329d03f07a0f54d5015d8b49dce2f24a77af",
    "pjainfo.pas": "fc3c14eb4f232740818d04fc94fcf9e65d901009",
    "pjainfoj2k.pas": "14a9a4ab351bd242dfbb8e1c1c906c1768d72fad",
    "pjainfojpeg.pas": "8f30e62a67aa2333ca48a1fe5d50928d3eba7d52",
    "pjainfopja.pas": "9247741bafa3b5dd338cf045c2a0f67e0d636647",
    "pjaj2k.pas": "77477c9aa6c3132fc445ea48b8b00ac6bbf9194b",
    "pjaj2kcs.pas": "30e085bbaec4f7c19ecfec6e716aa603d1cc275c",
    "pjaj2khm.c": "86fb1efdb89bf2aefe485cc33688b81658eb374b",
    "pjaj2kht.c": "7c06d04ee3f263dfb01943daadcfb54ecf9fdb96",
    "pjaj2kht_tab.h": "274f91ec16e539baf12439c34e42aedc398bad0b",
    "pjaj2km.pas": "dc38483728592b837d05ac7b421ad22e27ffc51f",
    "pjaj2km_tab.h": "16dba82cf3457d805a2e01d501f01246598a20e3",
    "pjaj2kmc.c": "b19ea891d7665a65c6278cf224170cd7eb538665",
    "pjaj2kpm.c": "9b29a86d4380ef6e556c40c36acfdf7e9bcafb48",
    "pjaj2kpx.pas": "9ec3d7bb41d833bf4cc95bd33c589e2781e51f87",
    "pjaj2kt1.c": "d43834c4086d688461cd5779dc50ec302a539ab9",
    "pjaj2kt1.h": "2443167f3f9dd19d0391d1337ec637d0a24da810",
    "pjajls.pas": "6f086063a4b7220a5c8c557a335e1b2360097500",
    "pjajpeg.pas": "b323ccdae5b72b4e1bfc7a5bcc27b5b69e5297eb",
    "pjalimits.pas": "c69b6d2264e6c022f5d226b0744d6e8ba22eb70b",
    "pjamodel.pas": "318cccb5b6a3aa26fb334fcf589e432fb6581b2a",
    "pjapack.pas": "7869f02df28aefe9dce6526835bcc8aca1963547",
    "pjareport.pas": "0c946432fbbc44eb7b3a19510488351991081385",
    "pjaselftest.inc": "8bc7b3690e24a4faf32b48fbaa57785edcef4951",
    "pjaselftest.pas": "8d8b01bfa7428af6c797b00edd81575cf4569426",
    "pjastage.pas": "66395d9de6cd7f5f26d78e56cfedb07a23f560c3",
    "pjasysc.c": "70645a81ac48e8585413f4e41a571f2c1c2157a5",
    "pjasysinfo.pas": "6616861baaeed99f4fbf1311f58301df256539d7",
    "pjathread.pas": "d0062434a27cad319655a76d7ced318daf4edc53",
    "pjav7.pas": "1db4d4f72e4a3085d7549bf08e73918dac80e5c9",
    "pjav7c.c": "b00f5338596c61a72b4ef631a08d357629377af0",
    "pjav7c_loop.h": "73ebad12da5a1cc96738dda997b78661f2be5235",
}


def git_blob(data):
    return sha1(b"blob " + str(len(data)).encode("ascii") + b"\0" + data).hexdigest()


def main():
    destination = Path(__file__).resolve().parents[1] / "out/pja-port-20261117/source"
    # Inspect all existing paths before making any changes.
    for name, expected in BLOBS.items():
        path = destination / name
        if path.exists() and git_blob(path.read_bytes()) != expected:
            raise RuntimeError(f"Existing source differs from pinned Git blob; refusing to overwrite: {path}")
    destination.mkdir(parents=True, exist_ok=True)
    for name, expected in BLOBS.items():
        path = destination / name
        if path.exists():
            print(f"Verified existing {name}: {expected}")
            continue
        url = f"https://raw.githubusercontent.com/YadeWira/PJA/{REVISION}/pas/{name}"
        request = Request(url, headers={"User-Agent": "rs-jxl-plus-development-oracle"})
        with urlopen(request, timeout=60) as response:
            data = response.read(8 * 1024 * 1024 + 1)
        if git_blob(data) != expected:
            raise RuntimeError(f"Downloaded {name} differs from pinned Git blob {expected}")
        # Exclusive creation also protects a file appearing after the preflight.
        with path.open("xb") as output:
            output.write(data)
        print(f"Fetched and verified {name}: {expected}")


if __name__ == "__main__":
    try:
        main()
    except (OSError, RuntimeError) as error:
        print(error, file=sys.stderr)
        sys.exit(1)
