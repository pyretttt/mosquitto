// Tiny k6 load test. Run with `k6 run infra/load/k6.js`.
// Expects an image at tests/fixtures/cat.jpg.
import http from 'k6/http';
import { check, sleep } from 'k6';

export const options = {
  stages: [
    { duration: '30s', target: 5 },
    { duration: '1m', target: 10 },
    { duration: '30s', target: 0 },
  ],
  thresholds: {
    http_req_failed: ['rate<0.02'],
    http_req_duration: ['p(95)<800'],
  },
};

const image = open('../../tests/fixtures/cat.jpg', 'b');

export default function () {
  const res = http.post('http://localhost:8000/predict', {
    file: http.file(image, 'cat.jpg', 'image/jpeg'),
  });
  check(res, {
    'is 200': (r) => r.status === 200,
    'has top_class': (r) => r.json('top_class.label') !== undefined,
  });
  sleep(0.1);
}
