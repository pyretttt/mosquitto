terraform {
  backend "oci" {
    namespace = "ax871f0napcp"
    bucket    = "tofu-states"
    key       = "infra/terraform.tfstate"
  }
}