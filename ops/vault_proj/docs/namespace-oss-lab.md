Each team can have namespace with policies applied for example:

```
path "secret/data/team-a/*" {
  capabilities = ["read", "create", "update"]
}
path "secret/metadata/team-a/*" {
  capabilities = ["read", "list"]
}

path "secret/data/team-b/*" {
  capabilities = ["read", "create", "update"]
}
path "secret/metadata/team-b/*" {
  capabilities = ["read", "list"]
}
```

Uploading policies and issuing token for each team with

```
vault policy write team-a local/policies/team-a.hcl
vault token create -policy team-a -ttl=1h
```

Then secrets can be written/updated with
```
VAULT_TOKEN=${TEAM_A} vault kv put secret/team-a/cred ...
```

Any attempt to read/update secrets of other team fail

```
# fails
VAULT_TOKEN=${TEAM_A} vault kv put secret/team-b/cred ...
```
