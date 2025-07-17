import torch
from torch import nn
from torch.nn import functional as F
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

class Causal_HMM(nn.Module):
    def __init__(self, args, z_size, v_size, s_size, A_size, B_size, batch_size, layer_count=3, channels=3):
        super(Causal_HMM, self).__init__()

        d = args.image_size
        self.d = d
        self.z_size = z_size
        self.s_size = s_size
        self.v_size = v_size
        self.batch_size = batch_size
        self.args = args

        self.h_dim = z_size + v_size + s_size

        self.dim = int(d / (2 ** layer_count))

        # image encoder
        self.layer_count = layer_count

        inputs = channels
        mul = 1
        for i in range(self.layer_count):
            setattr(self, "post_conv%d" % (i + 1), nn.Conv2d(inputs, d * mul, 3, 1, 1))
            setattr(self, "post_conv%d_bn" % (i + 1), nn.BatchNorm2d(d * mul))
            inputs = d * mul
            mul *= 2

        inputs2 = channels
        mul2 = 1
        for i in range(self.layer_count):
            setattr(self, "post_conv2%d" % (i + 1), nn.Conv2d(inputs2, d * mul2, (1,9), 1, 1))
            setattr(self, "post_conv2%d_bn" % (i + 1), nn.BatchNorm2d(d * mul2))
            inputs2 = d * mul2
            mul2 *= 2

        inputs3 = channels
        mul3 = 1
        for i in range(self.layer_count):
            setattr(self, "post_conv3%d" % (i + 1), nn.Conv2d(inputs3, d * mul3, (11,1), 1, 1))
            setattr(self, "post_conv3%d_bn" % (i + 1), nn.BatchNorm2d(d * mul3))
            inputs3 = d * mul3
            mul3 *= 2
        self.d_max = inputs*3

        # image decoder
        self.d1 = nn.Linear(self.h_dim, 3*inputs * self.dim * self.dim)

        # A decoder
        self.fc_decode_A = nn.Linear(v_size, A_size)

        # prior encoder for B_t, h_t
        b_feature_dim = args.fc_dim
        self.fc_b_layer = nn.Linear(B_size, b_feature_dim)

        self.prior_gru_z = nn.GRUCell(self.z_size + b_feature_dim, self.args.lstm_out)
        self.prior_gru_z.bias_ih.data.fill_(0)
        self.prior_gru_z.bias_hh.data.fill_(0)

        self.prior_gru_s = nn.GRUCell(self.s_size + b_feature_dim, self.args.lstm_out)
        self.prior_gru_s.bias_ih.data.fill_(0)
        self.prior_gru_s.bias_hh.data.fill_(0)

        self.prior_gru_v = nn.GRUCell(self.v_size + b_feature_dim, self.args.lstm_out)
        self.prior_gru_v.bias_ih.data.fill_(0)
        self.prior_gru_v.bias_hh.data.fill_(0)

        # prior encoder for mu, logvar
        self.fc_z_mu = nn.Linear(self.args.lstm_out, self.z_size)
        self.fc_z_logvar = nn.Linear(self.args.lstm_out, self.z_size)

        self.fc_s_mu = nn.Linear(self.args.lstm_out, self.s_size)
        self.fc_s_logvar = nn.Linear(self.args.lstm_out, self.s_size)

        self.fc_v_mu = nn.Linear(self.args.lstm_out, self.v_size)
        self.fc_v_logvar = nn.Linear(self.args.lstm_out, self.v_size)

        # post encoder
        x_feature_dim = self.d_max
        a_feature_dim = args.fc_dim

        self.post_encode_A = nn.Linear(A_size, a_feature_dim)
        self.post_encode_B = nn.Linear(B_size, b_feature_dim)

        self.post_z_fc = nn.Linear(x_feature_dim + b_feature_dim + self.z_size, self.args.fc_dim)
        self.post_z1 = nn.Linear(self.args.fc_dim, self.z_size)
        self.post_z2 = nn.Linear(self.args.fc_dim, self.z_size)

        self.post_v_fc = nn.Linear(x_feature_dim + a_feature_dim + b_feature_dim + self.v_size, self.args.fc_dim)
        self.post_v1 = nn.Linear(self.args.fc_dim, self.v_size)
        self.post_v2 = nn.Linear(self.args.fc_dim, self.v_size)

        self.post_s_fc = nn.Linear(x_feature_dim + a_feature_dim + b_feature_dim + self.s_size, self.args.fc_dim)
        self.post_s1 = nn.Linear(self.args.fc_dim, self.s_size)
        self.post_s2 = nn.Linear(self.args.fc_dim, self.s_size)

        mul = 3*inputs // d // 2
        inputs=inputs*3
        for i in range(1, self.layer_count):
            setattr(self, "deconv%d" % (i + 1), nn.ConvTranspose2d(inputs, d * mul, 4, 2, 1))
            setattr(self, "deconv%d_bn" % (i + 1), nn.BatchNorm2d(d * mul))

            inputs = d * mul
            mul //= 2

        setattr(self, "deconv%d" % (self.layer_count + 1), nn.ConvTranspose2d(inputs, channels, 4, 2, 1))


    def prior_encode(self, B_t_last, z_t_last, s_t_last, v_t_last):

        b = self.fc_b_layer(B_t_last)

        z_gru_input = torch.cat((b, z_t_last), 1)
        s_gru_input = torch.cat((b, s_t_last), 1)
        v_gru_input = torch.cat((b, v_t_last), 1)

        z_t = self.prior_gru_z(z_gru_input)
        s_t = self.prior_gru_s(s_gru_input)
        v_t = self.prior_gru_v(v_gru_input)

        mu_z = self.fc_z_mu(z_t)
        logvar_z = self.fc_z_logvar(z_t)

        mu_s = self.fc_s_mu(s_t)
        logvar_s = self.fc_s_logvar(s_t)

        mu_v = self.fc_v_mu(v_t)
        logvar_v = self.fc_v_logvar(v_t)

        return mu_z, logvar_z, mu_s, logvar_s, mu_v, logvar_v


    def post_encode(self, x_t, A_t, B_t_last, z_t_last, s_t_last, v_t_last):
        x_t2=x_t
        x_t3=x_t
        target_size = (128,128)
        pad_width = (0, 0, 1, 1)
        for i in range(self.layer_count):
            x_t = F.relu(getattr(self, "post_conv%d_bn" % (i + 1))(getattr(self, "post_conv%d" % (i + 1))(x_t)))
            x_t = F.dropout(x_t, p=0.5, training=self.training)
        for i in range(self.layer_count):
            x_t2 = F.relu(getattr(self, "post_conv2%d_bn" % (i + 1))(getattr(self, "post_conv2%d" % (i + 1))(x_t2)))
            x_t2 = F.dropout(x_t2, p=0.5, training=self.training)
        for i in range(self.layer_count):
            x_t3 = F.relu(getattr(self, "post_conv3%d_bn" % (i + 1))(getattr(self, "post_conv3%d" % (i + 1))(x_t3)))
            x_t3 = F.dropout(x_t3, p=0.5, training=self.training)
        x_t2 = torch.nn.functional.pad(x_t2, pad_width)
        x_t2 = torch.nn.functional.interpolate(x_t2, size=target_size, mode='bilinear',
                                                      align_corners=False)
        x_t3 = torch.nn.functional.pad(x_t3, pad_width)
        x_t3 = torch.nn.functional.interpolate(x_t3, size=target_size, mode='bilinear',
                                                      align_corners=False)
        # print(x_t.shape)
        # print(x_t2.shape)
        # print(x_t3.shape)
        # torch.Size([1, 2048, 4, 4])4
        # torch.Size([1, 2048, 138, 103])1,8
        # torch.Size([1, 2048, 103, 138])8,1
        x_t =torch.cat((x_t, x_t2, x_t3), 1)

        x_t = torch.nn.functional.adaptive_avg_pool2d(x_t, (1, 1))

        x_t = x_t.view(x_t.shape[0], self.d_max)
        feature_A = self.post_encode_A(A_t)
        feature_B = self.post_encode_B(B_t_last)

        f_z = torch.cat((x_t, feature_B, z_t_last), 1)
        f_z = self.post_z_fc(f_z)
        mu_z = self.post_z1(f_z)
        logvar_z = self.post_z2(f_z)

        f_s = torch.cat((x_t, feature_A, feature_B, s_t_last), 1)
        f_s = self.post_s_fc(f_s)
        mu_s = self.post_s1(f_s)
        logvar_s = self.post_s2(f_s)

        f_v = torch.cat((x_t, feature_A, feature_B, v_t_last), 1)
        f_v = self.post_v_fc(f_v)
        mu_v = self.post_v1(f_v)
        logvar_v = self.post_v2(f_v)

        return mu_z, logvar_z, mu_s, logvar_s, mu_v, logvar_v


    def reparameterize(self, mu, logvar, test):

        if not test:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)

            return eps.mul(std).add_(mu)
        else:

            return mu


    def decode_x(self, x):
        x = x.view(x.shape[0], self.h_dim)
        x = self.d1(x)
        x = x.view(x.shape[0], self.d_max, self.dim, self.dim)
        x = F.leaky_relu(x, 0.2)

        for i in range(1, self.layer_count):
            x = F.leaky_relu(getattr(self, "deconv%d_bn" % (i + 1))(getattr(self, "deconv%d" % (i + 1))(x)), 0.2)
        x = F.tanh(getattr(self, "deconv%d" % (self.layer_count + 1))(x))

        return x

    def decode_A(self, v_t):
        A = self.fc_decode_A(v_t)
        return A

    def forward(self, x_t, A_t, B_t_last, z_t_last, s_t_last, v_t_last, test=False):

        mu_z_prior, logvar_z_prior,\
        mu_s_prior, logvar_s_prior, mu_v_prior, logvar_v_prior = self.prior_encode(B_t_last, z_t_last, s_t_last, v_t_last)
        mu_z, logvar_z, mu_s, logvar_s, mu_v, logvar_v = self.post_encode(x_t, A_t, B_t_last, z_t_last, s_t_last, v_t_last)

        mu_h = torch.cat((mu_z, mu_s, mu_v), 1)
        logvar_h = torch.cat((logvar_z, logvar_s, logvar_v), 1)

        mu_prior = torch.cat((mu_z_prior, mu_s_prior, mu_v_prior), 1)
        logvar_prior = torch.cat((logvar_z_prior, logvar_s_prior, logvar_v_prior), 1)

        h_t = self.reparameterize(mu_h, logvar_h, test)
        v_t = self.reparameterize(mu_v, logvar_v, test)

        vae_rec_x_t = self.decode_x(h_t.view(-1, self.h_dim, 1, 1))
        vae_rec_A = self.decode_A(v_t)

        return vae_rec_x_t, vae_rec_A, mu_h, logvar_h, \
               mu_prior, logvar_prior, mu_z, mu_s, mu_v, logvar_v, mu_v_prior, logvar_v_prior

    def _init_papameters(self, args):

        # initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                if 'kaiming' in args.init:
                    nn.init.kaiming_normal(m.weight)
                    nn.init.constant_(m.bias, 0)
                elif 'xavier' in args.init:
                    nn.init.xavier_normal_(m.weight)
                    nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.GRUCell):
                if 'kaiming' in args.init:
                    nn.init.kaiming_normal(m.weight_hh.data)
                    nn.init.kaiming_normal(m.weight_ih.data)
                    nn.init.constant_(m.bias_ih.data, 0)
                    nn.init.constant_(m.bias_hh.data, 0)
                elif 'xavier' in args.init:
                    nn.init.xavier_normal_(m.weight_hh.data)
                    nn.init.xavier_normal_(m.weight_ih.data)
                    nn.init.constant_(m.bias_ih.data, 0)
                    nn.init.constant_(m.bias_hh.data, 0)


class Disease_Classifier(nn.Module):
    def __init__(self, args, in_dim):
        super(Disease_Classifier, self).__init__()

        self.fc1 = nn.Linear(in_dim, args.cls_fc_dim)
        self.fc2 = nn.Linear(args.cls_fc_dim, 2)

    def forward(self, x):

        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x
# import pickle
# import joblib 
# from sklearn.decomposition import PCA
# class Disease_Classifier(nn.Module):
#     def __init__(self, args, in_dim):
#         super(Disease_Classifier, self).__init__()
#         # Load the trained SVM model
#         self.svm_model = joblib.load('svm_model.pkl')
        
#         # Initialize PCA for dimensionality reduction
#         self.pca = PCA(n_components=15)
        
#         # Define the fully connected layers
#         self.fc1 = nn.Linear(in_dim + 2, args.cls_fc_dim)  # Adding 2 for SVM outputs
#         self.fc2 = nn.Linear(args.cls_fc_dim, 2)  # Assuming binary classification

#     def forward(self, x):
#         # Convert input features to numpy array
#         x_np = x.detach().cpu().numpy()
        
#         # Apply PCA to reduce dimensionality
#         x_np_reduced = self.pca.fit_transform(x_np)
        
#         # Get SVM predictions
#         svm_predictions = self.svm_model.predict(x_np_reduced)
        
#         # Convert SVM predictions to tensor
#         svm_predictions = torch.tensor(svm_predictions, dtype=torch.float32).unsqueeze(1)  # Shape (batch_size, 1)

#         # For binary classification, you may want to convert predictions to probabilities
#         svm_probabilities = self.svm_model.predict_proba(x_np_reduced)
#         svm_probabilities = torch.tensor(svm_probabilities, dtype=torch.float32)  # Shape (batch_size, num_classes)

#         # Combine SVM output with input
#         combined_input = torch.cat((x, svm_probabilities), dim=1)

#         # Pass through fully connected layers
#         x = F.relu(self.fc1(combined_input))
#         x = self.fc2(x)

#         return x