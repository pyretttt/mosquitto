lakefs() {
    if [ "$1" == "start" ]; then
        export MOUNT_FS="/mnt/data_storage"
        MOUNT_INFO=$(mount | grep "$MOUNT_FS")
        if [ -n "$MOUNT_INFO" ]; then
            USER_ID=1000
            GROUP_ID=1000

            # List of directories to create
            DIRS=(
                "/mnt/data_storage/opt/lakefs/metadata"
                "/mnt/data_storage/opt/lakefs/minio_data"
                "/mnt/data_storage/opt/lakefs/pgdata"
                "/mnt/data_storage/opt/lakefs/lakefs"
            )

            for dir in "${DIRS[@]}"; do
                if [ ! -d "$dir" ]; then
                    echo "Creating directory $dir"
                    sudo mkdir -p "$dir"
                else
                    echo "Directory $dir already exists"
                fi

                echo "Setting ownership to UID:$USER_ID GID:$GROUP_ID for $dir"
                sudo chown -R "$USER_ID":"$GROUP_ID" "$dir"
            done

            echo "Using external data mount $MOUNT_INFO"
            export LAKEFS_VOLUME="$MOUNT_FS/opt/lakefs"
        else
            export LAKEFS_VOLUME="/opt/lakefs"
        fi
        cd ~/dev/mosquitto/ml/infra/lakefs && docker compose up -d --build
    elif [ "$1" == "stop" ]; then
        cd ~/dev/mosquitto/ml/infra/lakefs && docker compose down
    else
        echo "Command not specified"
    fi
}
