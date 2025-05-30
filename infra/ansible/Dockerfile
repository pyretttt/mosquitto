FROM ansible/ansible:latest

ARG HOST
ARG USER
ARG PASS

WORKDIR /work
COPY playbook.yaml .

RUN ansible-playbook playbook.yaml \
    -i "${HOST}" \
    -u "${USER}" \
    --extra-vars "ansible_ssh_pass=${PASS}" \
    --ssh-extra-args='-o StrictHostKeyChecking=no'