import os
import synapseclient
from synapseclient.operations import download_list_files

DOWNLOAD_DIR = os.path.expanduser("~/datasets/BraTS2024")

os.makedirs(DOWNLOAD_DIR, exist_ok=True)

token = os.environ.get("SYNAPSE_AUTH_TOKEN")
if not token:
    raise RuntimeError("SYNAPSE_AUTH_TOKEN is not set")

syn = synapseclient.Synapse()
syn.login(authToken=token)

manifest_path = download_list_files(
    download_location=DOWNLOAD_DIR,
    parallel=False,
    synapse_client=syn,
)

print("Download pass finished.")
print("Manifest:", manifest_path)
