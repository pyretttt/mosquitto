provider "kubernetes" {
  config_path = var.k8s_config_file
}
provider "oci" {
  region = var.region
}

data "oci_containerengine_node_pool" "k8s_np" {
  node_pool_id = var.node_pool_id
}

locals {
  active_nodes = [for node in data.oci_containerengine_node_pool.k8s_np.nodes : node if node.state == "ACTIVE"]
}

resource "kubernetes_namespace" "core_namespace" {
  metadata {
    name = "core-ns"
  }
}

resource "kubernetes_deployment_v1" "nginx_deployment" {
  metadata {
    name = "nginx"
    labels = {
      app = "nginx"
    }
    namespace = kubernetes_namespace.core_namespace.id
  }
  spec {
    replicas = 1
    selector {
      match_labels = {
        app = "nginx"
      }
    }
    template {
      metadata {
        labels = {
          app = "nginx"
        }
      }
      spec {
        container {
          image = "docker.io/library/nginx:latest"
          name  = "nginx"
          port {
            container_port = 80
          }
        }
      }
    }
  }
}

# resource "kubernetes_service" "nginx_service" {
#   metadata {
#     name      = "nginx-service"
#     namespace = kubernetes_namespace.core_namespace.id
#   }
#   spec {
#     selector = {
#       app = "nginx"
#     }
#     port {
#       port        = 80
#       target_port = 80
#       node_port   = 31600
#     }
#     type = "NodePort"
#   }
# }
