There're a list of components implemented from scratch

Transformers:
- vanilla transformer is just vanilla attention + transformer layer. Also implements multihead attention
- transformer - is vanilla transfomer with cache above it.
- rotary_transformer - is cached transformer with rotary embeddings in between.

- Rotary Embedding is mostly pytorch implementation for single plain attention (not multihead)
- PositionEmbeggins - is sinusoidal absolute position transformer

- Encoder Decoder Transformer - is seq 2 seq transformer encoder decoder implementation