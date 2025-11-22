from sklearn.decomposition import PCA

def apply_pca(data, n_components=2):
    pca = PCA(n_components=n_components)
    principal_components = pca.fit_transform(data)
    return principal_components, pca.explained_variance_ratio_