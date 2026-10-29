# Planned data-pipeline terminology and responsibilities:
#
# - serialize_record(): create prompt and response strings for one extraction record.
# - encode_record(): tokenize one prompt/response pair, append one EOS token, and create
#   response-only labels and loss masks.
# - collate_batch(): pad encoded records and return batched model tensors and masks.
#
# Possible later additions:
# - validate_sequence_length()
# - ExtractionRecord and EncodedRecord data structures
# - ExtractionDataset
# - build_dataloader()
#
# Keep Synthea reading, patient splitting, and synthetic-note rendering in separate modules.
