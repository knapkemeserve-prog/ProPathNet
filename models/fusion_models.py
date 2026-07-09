import torch
import torch.nn as nn
import torch.nn.functional as F


class self_attention(nn.Module):
    def __init__(self, in_features, heads, dropout_prob=0):
        super(self_attention, self).__init__()
        self.in_features = in_features
        self.norm1 = nn.LayerNorm(sum(in_features))
        self.atten = nn.MultiheadAttention(sum(in_features), heads)
        self.norm2 = nn.LayerNorm(sum(in_features))
        self.residual = nn.Linear(sum(in_features), sum(in_features))
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, x):
        x = torch.cat(x, dim=1)
        x = self.atten(self.norm1(x), self.norm1(x), self.norm1(x))[0] + x
        x = self.residual(self.norm2(x)) + x
        x = self.dropout(x)
        x = torch.split(x, [i for i in self.in_features], dim=1)
        return x


class cross_attention(nn.Module):
    def __init__(self, in_features, heads, dropout_prob=0):
        super(cross_attention, self).__init__()
        self.in_features = in_features
        self.norm1 = nn.ModuleList([nn.LayerNorm(i) for i in in_features])
        self.atten = nn.ModuleList(
            [
                nn.MultiheadAttention(
                    i, heads, kdim=sum(in_features), vdim=sum(in_features)
                )
                for i in in_features
            ]
        )
        self.norm2 = nn.ModuleList([nn.LayerNorm(i) for i in in_features])
        self.residual = nn.ModuleList([nn.Linear(i, i) for i in in_features])
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, x):
        x_i = [self.norm1[i](x[i]) for i in range(len(x))]
        x_all = torch.cat(x, dim=1)
        x = [self.atten[i](x_i[i], x_all, x_all)[0] + x[i] for i in range(len(x))]
        x = [self.residual[i](self.norm2[i](x[i])) + x[i] for i in range(len(x))]
        x = [self.dropout(x[i]) for i in range(len(x))]
        return x


class bottleneck_cross_attention(nn.Module):
    def __init__(self, in_features, bottleneck=8, dropout_prob=0):
        super(bottleneck_cross_attention, self).__init__()
        self.in_features = in_features
        self.bottleneck = bottleneck
        self.norm = nn.ModuleList(
            [nn.LayerNorm(i + self.bottleneck) for i in in_features]
        )
        self.atten = nn.ModuleList(
            [
                nn.MultiheadAttention(
                    i + self.bottleneck,
                    1,
                    kdim=i + self.bottleneck,
                    vdim=i + self.bottleneck,
                )
                for i in in_features
            ]
        )
        self.residual = nn.ModuleList(
            [nn.Linear(i + self.bottleneck, i + self.bottleneck) for i in in_features]
        )
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, x, x_bottleneck=None):
        if x_bottleneck is None:
            x_bottleneck = torch.zeros((x[0].shape[0], self.bottleneck)).to(x[0].device)

        x = [torch.cat((x[i], x_bottleneck), dim=1) for i in range(len(x))]
        x = [self.norm[i](x[i]) for i in range(len(x))]

        x = [self.atten[i](x[i], x[i], x[i])[0] + x[i] for i in range(len(x))]
        x = [self.residual[i](self.norm[i](x[i])) + x[i] for i in range(len(x))]

        x_bottleneck = torch.mean(
            torch.stack([x[i][:, -self.bottleneck :] for i in range(len(x))]), dim=0
        )
        x = [x[i][:, : -self.bottleneck] for i in range(len(x))]

        x = [self.dropout(x[i]) for i in range(len(x))]

        return x, x_bottleneck


class sub_model(nn.Module):
    # a wrapper for a sub model
    def __init__(self, model, freeze=True, resize_latent_space=0):
        super(sub_model, self).__init__()
        self.model = model
        self.freeze = freeze
        if self.freeze:
            for param in self.model.parameters():
                param.requires_grad = False

        if resize_latent_space > 0:
            self.latent_size = resize_latent_space
            # self.model.out = nn.Linear(self.model.out.in_features, self.latent_size)
            self.model.out = nn.Sequential(
                nn.Linear(self.model.out.in_features, self.latent_size),
                nn.LayerNorm(self.latent_size),
                nn.ReLU(),
                nn.Dropout(0.2),
            )
            if hasattr(self.model, "mlp_head"):
                self.model.mlp_head[1] = self.model.out
        else:
            self.latent_size = self.model.out.in_features
            self.model.out = nn.Identity()
            if hasattr(self.model, "mlp_head"):
                self.model.mlp_head[1] = nn.Identity()

    def forward(self, x):
        x = self.model(x)
        return x


class fusion_transformer(nn.Module):
    def __init__(
        self,
        sub_models,
        num_fusion_layers=1,
        fusion_type="self",
        heads=8,
        fusion_dim=8,
        dropout_prob=0.0,
        n_classes=400,
        freeze_submodels=True,
        resize_latent_space=None,
    ):
        super(fusion_transformer, self).__init__()

        self.freeze_submodels = freeze_submodels
        self.n_models = len(sub_models)
        for i, model in enumerate(sub_models):
            if resize_latent_space is not None:
                self.add_module(
                    "sub_model{}".format(i),
                    sub_model(
                        model,
                        freeze=self.freeze_submodels,
                        resize_latent_space=resize_latent_space[i],
                    ),
                )
            else:
                self.add_module(
                    "sub_model{}".format(i),
                    sub_model(model, freeze=self.freeze_submodels),
                )

        self.fusion_type = fusion_type
        self.n_fusion_layers = num_fusion_layers
        for i in range(self.n_fusion_layers):
            self.add_module(
                "fusion_layer{}".format(i),
                self.make_fusion_layer(
                    [
                        getattr(self, "sub_model{}".format(i)).latent_size
                        for i in range(self.n_models)
                    ],
                    fusion_type=self.fusion_type,
                    heads=heads,
                    fusion_dim=fusion_dim,
                    dropout_prob=dropout_prob,
                ),
            )

        self.latent_space_size = sum(
            [
                getattr(self, "sub_model{}".format(i)).latent_size
                for i in range(self.n_models)
            ]
        )
        self.dropout = nn.Dropout(dropout_prob)
        self.out = nn.Linear(self.latent_space_size, n_classes)

    def make_fusion_layer(
        self, in_features, fusion_type, heads, fusion_dim, dropout_prob
    ):
        assert fusion_type in [
            "self",
            "cross",
            "bottleneck",
        ], "fusion_type must be one of self, cross, or bottleneck"
        if fusion_type == "self":  # self-attention
            return self_attention(in_features, heads, dropout_prob=dropout_prob)
        elif fusion_type == "cross":  # cross-attention
            return cross_attention(in_features, heads, dropout_prob=dropout_prob)
        elif fusion_type == "bottleneck":  # bottleneck attention
            return bottleneck_cross_attention(
                in_features, bottleneck=fusion_dim, dropout_prob=dropout_prob
            )

    def forward(self, x):
        latent_spaces = [
            getattr(self, "sub_model{}".format(i))(x[i]) for i in range(self.n_models)
        ]

        for i in range(self.n_fusion_layers):
            if self.fusion_type == "bottleneck":
                if i == 0:
                    bottleneck = None
                latent_spaces, bottleneck = getattr(self, "fusion_layer{}".format(i))(
                    latent_spaces, bottleneck
                )
            else:
                latent_spaces = getattr(self, "fusion_layer{}".format(i))(latent_spaces)

        x = torch.cat(latent_spaces, dim=1)
        x = self.dropout(x)
        x = self.out(x)
        return x

    def load_submodel(self, i, path):
        submodel = torch.load(path, map_location="cpu")

        # don't load the last layer
        submodel["state_dict"].pop("out.weight")
        submodel["state_dict"].pop("out.bias")
        try:  # don't load the mlp head if it exists
            # submodel['state_dict'].pop('mlp_head.0.weight')
            # submodel['state_dict'].pop('mlp_head.0.bias')
            submodel["state_dict"].pop("mlp_head.1.weight")
            submodel["state_dict"].pop("mlp_head.1.bias")
        except:
            pass
        getattr(self, "sub_model{}".format(i)).model.load_state_dict(
            submodel["state_dict"], strict=False
        )


class fusion_mlp(nn.Module):
    def __init__(
        self,
        sub_models,
        num_fusion_layers=1,
        fusion_dim=8,
        dropout_prob=0.0,
        n_classes=400,
        freeze_submodels=True,
        resize_latent_space=None,
    ):
        super(fusion_mlp, self).__init__()

        self.freeze_submodels = freeze_submodels
        self.n_models = len(sub_models)
        for i, model in enumerate(sub_models):
            if resize_latent_space is not None:
                self.add_module(
                    "sub_model{}".format(i),
                    sub_model(
                        model,
                        freeze=self.freeze_submodels,
                        resize_latent_space=resize_latent_space[i],
                    ),
                )
            else:
                self.add_module(
                    "sub_model{}".format(i),
                    sub_model(model, freeze=self.freeze_submodels),
                )

        self.latent_space_size = sum(
            [
                getattr(self, "sub_model{}".format(i)).latent_size
                for i in range(self.n_models)
            ]
        )
        self.n_fusion_layers = num_fusion_layers
        if self.n_fusion_layers > 1:
            self.fusion_layers = nn.ModuleList(
                [nn.Linear(self.latent_space_size, fusion_dim)]
            )
            for i in range(1, self.n_fusion_layers - 1):
                self.fusion_layers.append(nn.Linear(fusion_dim, fusion_dim))
            self.out = nn.Linear(fusion_dim, n_classes)
        elif self.n_fusion_layers == 1:
            self.out = nn.Linear(self.latent_space_size, n_classes)
        else:
            raise ValueError("num_fusion_layers must be at least 1")

        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, x):
        latent_spaces = [
            getattr(self, "sub_model{}".format(i))(x[i]) for i in range(self.n_models)
        ]
        x = torch.cat(latent_spaces, dim=1)
        if self.n_fusion_layers > 1:
            for i in range(self.n_fusion_layers - 1):
                x = self.dropout(x)
                x = F.relu(self.fusion_layers[i](x))

        x = self.out(x)
        return x

    def load_submodel(self, i, path):
        submodel = torch.load(path, map_location="cpu")

        # don't load the last layer
        submodel["state_dict"].pop("out.weight")
        submodel["state_dict"].pop("out.bias")
        try:  # don't load the mlp head if it exists
            # submodel['state_dict'].pop('mlp_head.0.weight')
            # submodel['state_dict'].pop('mlp_head.0.bias')
            submodel["state_dict"].pop("mlp_head.1.weight")
            submodel["state_dict"].pop("mlp_head.1.bias")
        except:
            pass
        getattr(self, "sub_model{}".format(i)).model.load_state_dict(
            submodel["state_dict"], strict=False
        )
