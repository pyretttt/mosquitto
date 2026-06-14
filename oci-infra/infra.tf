provider "oci" {
  region = var.region
}

# Virtual Cloud Network (VCN)
# with 10.0.0.0/16 + /24 subnets
# 10  .  0  .  1  .  42
# │   │     │     │
# │   │     │     └── host within subnet (8 bits → 256 addresses)
# │   │     └───────  subnet ID (which /24 block)
# │   └────────────  VCN prefix (shared by all subnets)
# └─────────────────  VCN prefix
module "vcn" {
  source  = "oracle-terraform-modules/vcn/oci"
  version = "3.6.0"

  compartment_id = var.compartment_id
  region         = var.region

  internet_gateway_route_rules = null
  local_peering_gateways       = null
  nat_gateway_route_rules      = null

  vcn_name      = "k8s-vcn"
  vcn_dns_label = "k8svcn"
  vcn_cidrs     = ["10.0.0.0/16"]

  # An Internet Gateway (IGW) — lets public subnets reach the public internet (bidirectional: inbound + outbound).
  # A route table with a default rule: 0.0.0.0/0 → Internet Gateway.
  create_internet_gateway = true
  # A NAT Gateway (NAT GW) — lets private subnets initiate outbound connections to the internet, but not receive unsolicited inbound traffic from the internet.
  create_nat_gateway = true
  # An LPG lets two VCNs in the same region talk to each other privately, as if they were one network (no public internet involved). Each map key becomes the LPG display name.
  create_service_gateway = true
}

resource "oci_core_security_list" "private_subnet_sl" {
  compartment_id = var.compartment_id
  vcn_id         = module.vcn.vcn_id
  display_name   = "k8s-private-subnet-sl"

  # subnet-level virtual firewall in OCI:
  # Rules for outbound traffic leaving instances in subnets that use this security list.
  egress_security_rules {
    stateless        = false
    destination      = "0.0.0.0/0" # allows all outbound traffic to any destination
    destination_type = "CIDR_BLOCK"
    protocol         = "all" # any protocol
  }

  # Rules for inbound traffic arriving at instances in those subnets.
  ingress_security_rules {
    stateless   = false
    source      = "10.0.0.0/16" # inside this vcn, traffic from public internet is not allowed
    source_type = "CIDR_BLOCK"
    protocol    = "all" # allow all protocols
  }

  ingress_security_rules {
    stateless   = false
    source      = "10.0.0.0/24"
    source_type = "CIDR_BLOCK"
    protocol    = "6"
    tcp_options { # health checking the cluster from the Network Load Balancer
      min = 10256
      max = 10256
    }
  }
  ingress_security_rules {
    stateless   = false
    source      = "10.0.0.0/24"
    source_type = "CIDR_BLOCK"
    protocol    = "6"
    tcp_options {
      min = 31600
      max = 31600 # NodePort
    }
  }
}

resource "oci_core_security_list" "public_subnet_sl" {
  compartment_id = var.compartment_id
  vcn_id         = module.vcn.vcn_id
  display_name   = "k8s-public-subnet-sl"
  egress_security_rules {
    stateless        = false
    destination      = "0.0.0.0/0" # all outbound traffic is allowed
    destination_type = "CIDR_BLOCK"
    protocol         = "all" # allow all protocols
  }
  egress_security_rules {
    stateless        = false
    destination      = "10.0.1.0/24"
    destination_type = "CIDR_BLOCK"
    protocol         = "6"
    tcp_options {
      min = 31600 # NodePort
      max = 31600
    }
  }
  egress_security_rules {
    stateless        = false
    destination      = "10.0.1.0/24"
    destination_type = "CIDR_BLOCK"
    protocol         = "6"
    tcp_options {
      min = 10256 # health checking the cluster from the Network Load Balancer
      max = 10256
    }
  }

  ingress_security_rules {
    stateless   = false
    source      = "10.0.0.0/16" # allows vcn inbound traffic
    source_type = "CIDR_BLOCK"
    protocol    = "all"
  }

  ingress_security_rules {
    stateless   = false
    source      = "0.0.0.0/0" # allows all inbound traffic
    source_type = "CIDR_BLOCK"
    protocol    = "6"
    tcp_options { # for tcp port 6443
      min = 6443
      max = 6443
    }
  }
  ingress_security_rules {
    protocol    = "6"
    source      = "0.0.0.0/0"
    source_type = "CIDR_BLOCK"
    stateless   = false
    tcp_options {
      max = 80 # for load balancing to be able to comminicate with the public subnet
      min = 80
    }
  }
}

# Create private subnet in vcn
resource "oci_core_subnet" "vcn_private_subnet" {
  compartment_id             = var.compartment_id
  vcn_id                     = module.vcn.vcn_id
  cidr_block                 = "10.0.1.0/24" # 256 addresses in the subnet
  route_table_id             = module.vcn.nat_route_id
  security_list_ids          = [oci_core_security_list.private_subnet_sl.id]
  display_name               = "k8s-private-subnet"
  prohibit_public_ip_on_vnic = true
}

# Create public subnet in vcn
resource "oci_core_subnet" "vcn_public_subnet" {
  compartment_id    = var.compartment_id
  vcn_id            = module.vcn.vcn_id
  cidr_block        = "10.0.0.0/24" # 256 addresses in the subnet
  route_table_id    = module.vcn.ig_route_id
  security_list_ids = [oci_core_security_list.public_subnet_sl.id]
  display_name      = "k8s-public-subnet"
}

# Kubernetes Cluster
resource "oci_containerengine_cluster" "k8s_cluster" {
  compartment_id     = var.compartment_id
  kubernetes_version = "v1.34.2"
  name               = "k8s-cluster"
  vcn_id             = module.vcn.vcn_id
  endpoint_config {
    is_public_ip_enabled = true                                 # allows public access to the cluster
    subnet_id            = oci_core_subnet.vcn_public_subnet.id # public subnet
  }
  options {
    add_ons {
      is_kubernetes_dashboard_enabled = true
      is_tiller_enabled               = false
    }
    kubernetes_network_config {
      pods_cidr     = "10.244.0.0/16"
      services_cidr = "10.96.0.0/16"
    }
    service_lb_subnet_ids = [oci_core_subnet.vcn_public_subnet.id]
  }
}

data "oci_identity_availability_domains" "ads" {
  compartment_id = var.compartment_id
}

# define node pool for k8s cluster
resource "oci_containerengine_node_pool" "k8s_node_pool" {
  cluster_id         = oci_containerengine_cluster.k8s_cluster.id
  compartment_id     = var.compartment_id
  kubernetes_version = "v1.34.2"
  name               = "k8s-node-pool"
  node_config_details {
    placement_configs {
      availability_domain = data.oci_identity_availability_domains.ads.availability_domains[0].name
      subnet_id           = oci_core_subnet.vcn_private_subnet.id
    }
    size = 2 # 2 nodes in the node pool
  }
  node_shape = "VM.Standard.A1.Flex"
  node_shape_config {
    memory_in_gbs = 6
    ocpus         = 1
  }

  node_source_details {
    # Linux 7.9
    image_id    = "ocid1.image.oc1.me-dubai-1.aaaaaaaarqcig5soskrxcccj7u4l5uzbtcdl37kgksni746t6yoiwtinoasq"
    source_type = "image"
  }
  initial_node_labels {
    key   = "name"
    value = "k8s-cluster"
  }
  ssh_public_key = var.ssh_public_key
}
