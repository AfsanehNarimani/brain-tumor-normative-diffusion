# import os
# import synapseclient
# import synapseutils
#
# syn = synapseclient.Synapse()
# syn.login(authToken=os.environ["SYNAPSE_AUTH_TOKEN"])
#
# files = synapseutils.syncFromSynapse(
#     syn,
#     "syn60086071",
#     path=os.path.expanduser("~/datasets/BraTS2024")
# )
#
# print(f"Downloaded {len(files)} item(s)")
import os

import synapseclient
from synapseclient.operations import download_list_files

syn = synapseclient.Synapse()
syn.login(authToken=os.environ["SYNAPSE_AUTH_TOKEN"])

manifest_path = download_list_files(
    download_location=os.path.expanduser("~/datasets/BraTS2024"),
    parallel=False,
    synapse_client=syn,
)

print(manifest_path)