import random
import numpy as np
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class NoiseTransformation(object):
    def __init__(self, sigma):
        self.sigma = sigma

    def __call__(self, X):
        if X.device.type == 'cuda':
            X = X.cpu()
        noise = np.random.normal(loc=0, scale=self.sigma, size=X.shape)
        
        fft_window = np.fft.fft(X[:, 1:].numpy())
        noise = (np.random.randn(*fft_window.shape) + 1j * np.random.randn(*fft_window.shape)) * 0.5
        fft_window += noise
        ifft_window = np.fft.ifft(fft_window).real
        X[:, 1:] = torch.from_numpy(ifft_window)

        return X.to(device)
        # return torch.tensor(X.numpy() + noise, dtype=torch.float32, device=device)

class SubAnomaly(object):
    def __init__(self, portion_len):
        self.portion_len = portion_len

    def inject_frequency_anomaly(self, window,
                                 subsequence_length: int= None,
                                 compression_factor: int = None,
                                 scale_factor: float = None,
                                 trend_factor: float = None,
                                 shapelet_factor: bool = False,
                                 trend_end: bool = False,
                                 start_index: int = None
                                 ):

        window = window.clone()

        if subsequence_length is None:
            min_len = int(window.shape[0] * 0.1)
            max_len = int(window.shape[0] * 0.9)
            subsequence_length = np.random.randint(min_len, max_len)

        if compression_factor is None:
            compression_factor = np.random.randint(2, 5)

        if scale_factor is None:
            scale_factor = np.random.uniform(0.1, 2.0, window.shape[1])
            print('test')

        if start_index is None:
            start_index = np.random.randint(0, len(window) - subsequence_length)
        end_index = min(start_index + subsequence_length, window.shape[0])

        if trend_end:
            end_index = window.shape[0]

        anomalous_subsequence = window[start_index:end_index]

        anomalous_subsequence = anomalous_subsequence.repeat(compression_factor, 1)
        anomalous_subsequence = anomalous_subsequence[::compression_factor]

        anomalous_subsequence = anomalous_subsequence * scale_factor

        if trend_factor is None:
            trend_factor = np.random.normal(1, 0.5)
        coef = 1
        if np.random.uniform() < 0.5: coef = -1
        anomalous_subsequence = anomalous_subsequence + coef * trend_factor

        if shapelet_factor:
            anomalous_subsequence = window[start_index] + (torch.rand_like(window[start_index]) * 0.1)

        window[start_index:end_index] = anomalous_subsequence

        return np.squeeze(window)

    def __call__(self, X):
        window = X.clone()
        anomaly_seasonal = window.clone()
        anomaly_trend = window.clone()
        anomaly_global = window.clone()
        anomaly_contextual = window.clone()
        anomaly_shapelet = window.clone()
        min_len = int(window.shape[0] * 0.1)
        max_len = int(window.shape[0] * 0.9)
        subsequence_length = np.random.randint(min_len, max_len)
        start_index = np.random.randint(0, len(window) - subsequence_length)
        if (window.ndim > 1):
            num_features = window.shape[1]
            num_dims = np.random.randint(int(num_features/10), int(num_features/2))
            for k in range(num_dims):
                i = np.random.randint(0, num_features)
                temp_win = window[:, i].reshape((window.shape[0], 1))
                anomaly_seasonal[:, i] = self.inject_frequency_anomaly(temp_win,
                                                              scale_factor=1,
                                                              trend_factor=0,
                                                           subsequence_length=subsequence_length,
                                                           start_index = start_index)

                anomaly_trend[:, i] = self.inject_frequency_anomaly(temp_win,
                                                             compression_factor=1,
                                                             scale_factor=1,
                                                             trend_end=True,
                                                           subsequence_length=subsequence_length,
                                                           start_index = start_index)

                anomaly_global[:, i] = self.inject_frequency_anomaly(temp_win,
                                                            subsequence_length=2,
                                                            compression_factor=1,
                                                            scale_factor=8,
                                                            trend_factor=0,
                                                           start_index = start_index)

                anomaly_contextual[:, i] = self.inject_frequency_anomaly(temp_win,
                                                            subsequence_length=4,
                                                            compression_factor=1,
                                                            scale_factor=3,
                                                            trend_factor=0,
                                                           start_index = start_index)

                anomaly_shapelet[:, i] = self.inject_frequency_anomaly(temp_win,
                                                          compression_factor=1,
                                                          scale_factor=1,
                                                          trend_factor=0,
                                                          shapelet_factor=True,
                                                          subsequence_length=subsequence_length,
                                                          start_index = start_index)

        else:
            temp_win = window.reshape((len(window), 1))
            anomaly_seasonal = self.inject_frequency_anomaly(temp_win,
                                                          scale_factor=1,
                                                          trend_factor=0,
                                                          subsequence_length=subsequence_length,
                                                          start_index = start_index)

            anomaly_trend = self.inject_frequency_anomaly(temp_win,
                                                         compression_factor=1,
                                                         scale_factor=1,
                                                         trend_end=True,
                                                         subsequence_length=subsequence_length,
                                                         start_index = start_index)

            anomaly_global = self.inject_frequency_anomaly(temp_win,
                                                        subsequence_length=3,
                                                        compression_factor=1,
                                                        scale_factor=8,
                                                        trend_factor=0,
                                                        start_index = start_index)

            anomaly_contextual = self.inject_frequency_anomaly(temp_win,
                                                        subsequence_length=5,
                                                        compression_factor=1,
                                                        scale_factor=3,
                                                        trend_factor=0,
                                                        start_index = start_index)

            anomaly_shapelet = self.inject_frequency_anomaly(temp_win,
                                                      compression_factor=1,
                                                      scale_factor=1,
                                                      trend_factor=0,
                                                      shapelet_factor=True,
                                                      subsequence_length=subsequence_length,
                                                      start_index = start_index)

        anomalies = [anomaly_seasonal,
                     #anomaly_trend,
                     #anomaly_global,
                     #anomaly_contextual,
                     #anomaly_shapelet
                     ]

        anomalous_window = random.choice(anomalies)

        fft_window = np.fft.fft(window[:, 0].cpu().numpy())
        noise = (np.random.randn(len(fft_window)) + 1j * np.random.randn(len(fft_window))) * 0.5
        fft_window[:] += noise[:]
        ifft_window = np.fft.ifft(fft_window).real
        window[:, 0] = torch.from_numpy(ifft_window).to(device)

        return window
        # return anomalous_window
        # return temp_win