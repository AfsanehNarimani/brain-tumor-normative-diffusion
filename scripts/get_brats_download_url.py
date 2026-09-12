import os
import synapseclient

SYNAPSE_ID = "syn60086071"

syn = synapseclient.Synapse()
syn.login(authToken=os.environ["SYNAPSE_AUTH_TOKEN"])

entity = syn.get(SYNAPSE_ID, downloadFile=False)

file_handle_id = entity.dataFileHandleId

url = syn.restGET(
    f"/file/{file_handle_id}",
    endpoint=syn.fileHandleEndpoint,
    params={
        "redirect": "false",
        "fileAssociateType": "FileEntity",
        "fileAssociateId": SYNAPSE_ID,
    },
)

print(url)