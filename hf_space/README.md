---
title: Scentmate API
emoji: 🌸
colorFrom: pink
colorTo: yellow
sdk: docker
app_port: 7860
pinned: false
---

# Scentmate API

Personality-based fragrance recommender. A single ONNX model (`model/recommender.onnx`) holds the
trained trait encoder and the full catalogue of 41,704 fragrances, and does all the scoring,
filtering (gender, popularity, accord constraints) and ranking on CPU.

- `GET /` health check
- `GET /meta` traits, accords, note vocabulary
- `POST /recommend` see `/docs` for the request schema
- `POST /feedback` store a "like" for future retraining

Interactive docs: `/docs`
