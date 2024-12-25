from dyglib.models import DyGFormer

if __name__ == "__main__":
    
    dynamic_backbone = DyGFormer(
        node_raw_features=node_raw_features,
        edge_raw_features=edge_raw_features,
        neighbor_sampler=train_neighbor_sampler,
        time_feat_dim=args.time_feat_dim,
        channel_embedding_dim=args.channel_embedding_dim,
        patch_size=args.patch_size,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        dropout=args.dropout,
        max_input_sequence_length=args.max_input_sequence_length,
        device=args.device
    )
    