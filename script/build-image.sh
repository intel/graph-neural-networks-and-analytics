echo ""
echo -e "\nbuild docker DGL  image\n"
docker build -t gnn-wf . \
            --build-arg https_proxy=${https_proxy} \
            --build-arg HTTPS_PROXY=${HTTPS_PROXY} \
            --build-arg HTTP_PROXY=${HTTP_PROXY} \
            --build-arg http_proxy=${http_proxy} \
