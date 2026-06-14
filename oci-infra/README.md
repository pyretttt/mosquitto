# OCI infra
***

Basically all infra provisioned inside root folder. To update provisioning set up variables:
- [compartment_id](https://cloud.oracle.com/tenancy) is OCID field
- [region](https://cloud.oracle.com) upper right angle, now its `me-dubai-1`
- [ssh_public_key] - locally generate ssh key

What is provisioned:
1. VCN with /16 cid
2. Private and public subnets /24 with ingress and outgress rules
3. K8s container engine
4. Node pool with 2 VM.Standard.A1.Flex
5. Container registry. Container registry is - https://dxb.ocir.io (from https://docs.oracle.com/en-us/iaas/Content/Registry/Concepts/registryprerequisites.htm#Preparing_for_Registry). User name for this registry is composed of oci-user and [Object storage namespace](https://cloud.oracle.com/tenancy). User name is then <OBJECT_STORAGE_NAMESPACE>/<USER>. Password is obtained through adding AUTH_TOKEN at [tokens and keys](https://cloud.oracle.com/identity/domains/my-profile/auth-tokens) page.


To obtain K8s cluster config run `oci ce cluster create-kubeconfig --cluster-id <cluster_ocid> --file ~/.kube/oci-k8s-config --region <region> --token-version 2.0.0 --kube-endpoint PUBLIC_ENDPOINT`. Where `cluster_ocid` can be obtained from `terraform outputs`


# K8S
***

K8s creates root namespaces and load balancer everything else is up to k8s adminstration through kubectl.

1. Namespace is `core-ns`
2. Network load balancer has healthcheck at 10256 port egress. And 31600 is nodeport port. Internet traffic is through port 80.

To update provisioning set up variables:
1. [compartment_id](#oci-infra)
2. [region](#oci-infra)
3. public_subnet_id can be obtained from `cd oci-infra && terraform outputs`
4. node_pool_id can be obtained from `cd oci-infra && terraform outputs`
5. k8s_config_file is [locally stored provisioned k8s config](#oci-infra).