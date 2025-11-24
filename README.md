# LM4CP
LM sinh lời giải: 
chmod +x generator.sh
./generator.sh


Chạy trình chấm ioi:
cd run_tests

sudo docker rm -f piston_worker0

sudo docker run -d \
  --name piston_worker0 \
  -v "$PWD/piston_packages:/piston/packages" \
  -e PORT=2000 \
  -e PISTON_COMPILE_TIMEOUT=60000 \
  -e PISTON_RUN_TIMEOUT=60000 \
  -e PISTON_OUTPUT_MAX_SIZE=1000000000 \
  -e PISTON_MAX_FILE_SIZE=1000000000 \
  -e PISTON_DISABLE_NETWORKING=true \
  -e PISTON_REPO_URL=https://github.com/guipenedo/piston/releases/download/pkgs/index \
  -p 2000:2000 \
  --entrypoint /bin/bash \
  ghcr.io/engineer-man/piston@sha256:63b5654156a89c5a2ad281aface21416615d62ec056d88efe8fcd307ce73575a \
  -c "sed -i '/app.use(body_parser.urlencoded/c\  app.use(body_parser.urlencoded({ extended: true, limit: \"512mb\" }));' src/index.js && \
      sed -i '/app.use(body_parser.json/c\  app.use(body_parser.json({ limit: \"512mb\" }));' src/index.js && \
      node src"


curl -X POST http://localhost:2000/api/v2/packages \
  -H "Content-Type: application/json" \
  -d '{"language": "cms_ioi", "version": "1.0.0"}'


export PISTON_ENDPOINTS=http://localhost:2000/api/v2


python tests_runner.py \
  tiendung6b/qwen3-4b-instruct-2507-ioi_2024_4k \
  tiendung6b/qwen3-4b-instruct-2507-ioi_2024_4k-results \
  --local_results_path results \
  --max_concurrent_requests 1 \
  --test_batch_size 1 \
  --id_column uuid \
  --add_includes \
  --add_messages_column

