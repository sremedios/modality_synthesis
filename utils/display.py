from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns
from scipy.stats import entropy
from sklearn.neighbors import KernelDensity as KDE


def histogram_intersection(hist1, hist2):
    numerator = 0
    for a, b in zip(hist1, hist2):
        numerator += min(a, b)

    denominator = min(np.sum(hist1), np.sum(hist2))

    return numerator/denominator


def show_image(img_data):
    plt.imshow((img_data/np.max(img_data)*255).astype(np.uint8),
               interpolation='nearest',
               cmap="Greys")
    plt.show()


def compare_histograms(img_data_1, img_data_2, test_loc, ignore_zero=True):

    img_1 = (img_data_1/np.max(img_data_1) * 255).T.astype(np.uint8).flatten()
    img_2 = (img_data_2/np.max(img_data_2) * 255).T.astype(np.uint8).flatten()
    diff = np.abs((img_data_1/np.max(img_data_1) * 255).T - (img_data_2/np.max(img_data_2) *
                                                             255).T).astype(np.uint8).flatten()

    min_val = 0  # np.min(img_1)
    max_val = np.max(img_1)
    bins = np.linspace(min_val, max_val, 255)

    exclusion_threshold = 5
    if ignore_zero:
        img_1 = img_1[img_1 > exclusion_threshold]
        img_2 = img_2[img_2 > exclusion_threshold]
        diff = diff[diff > exclusion_threshold]

    '''
    plt.hist(img_1, bins, alpha=0.5, label="Original")
    plt.hist(img_2, bins, alpha=0.5, label="Reconstruction")
    plt.hist(diff, bins, alpha=0.5, label="Difference")

    sns.distplot(img_1, hist=False, kde=True, bins=bins, label="Original")
    sns.distplot(img_2, hist=False, kde=True, bins=bins, label="Reconstruction")
    sns.distplot(diff, hist=False, kde=True, bins=bins, label="Difference")
    '''

    num_samples = int(np.prod([x for x in img_1.shape if x is not None]))

    kde_orig = KDE(kernel='gaussian', bandwidth=0.75).fit(
        np.reshape(img_1, img_1.shape + (1,)))
    kde_orig_samples = kde_orig.sample(num_samples, random_state=0)
    plt.hist(kde_orig_samples, bins, alpha=0.5, label="kde_orig samples")
    plt.hist(img_1, bins, alpha=0.5, label="Original")

    kde_recon = KDE(kernel='gaussian', bandwidth=0.75).fit(
        np.reshape(img_2, img_2.shape + (1,)))
    kde_recon_samples = kde_recon.sample(num_samples, random_state=0)
    plt.hist(kde_recon_samples, bins, alpha=0.5, label="kde_recon samples")
    plt.hist(img_2, bins, alpha=0.5, label="Recon")

    '''
    kde_diff = KDE(kernel='gaussian', bandwidth=0.75).fit(np.reshape(diff, diff.shape + (1,)))
    kde_diff_samples = kde_diff.sample(num_samples, random_state=0)
    plt.hist(kde_diff_samples, bins, alpha=0.5, label="kde_diff samples")
    plt.hist(diff, bins, alpha=0.5, label="Diff")
    '''

    # require non-zero values to find KL divergence
    kde_orig_samples[kde_orig_samples <= 0] = 1e-9
    kde_recon_samples[kde_recon_samples <= 0] = 1e-9
    #kde_diff_samples[kde_diff_samples <= 0] = 1e-9

    orig_recon_kl_divergence = entropy(kde_orig_samples, kde_recon_samples)
    #orig_diff_kl_divergence = entropy(kde_orig_samples, kde_diff_samples)
    #recon_diff_kl_divergence = entropy(kde_recon_samples, kde_diff_samples)

    '''
    print("Orig Recon KLD:", orig_recon_kl_divergence)
    print("Orig Diff KLD:", orig_diff_kl_divergence)
    print("Recon Diff KLD:", recon_diff_kl_divergence)
    '''

    orig_recon_hist_inter = histogram_intersection(
        kde_orig_samples, kde_recon_samples)
    #orig_diff_hist_inter = histogram_intersection(kde_orig_samples, kde_diff_samples)
    #recon_diff_hist_inter = histogram_intersection(kde_recon_samples, kde_diff_samples)

    '''
    print("Orig Recon HI:", orig_recon_hist_inter)
    print("Orig Diff HI:", orig_diff_hist_inter)
    print("Recon Diff HI:", recon_diff_hist_inter)

    a = np.abs(orig_recon_hist_inter - orig_diff_hist_inter)
    b = np.abs(orig_recon_hist_inter - recon_diff_hist_inter)
    c = np.abs(recon_diff_hist_inter - orig_recon_hist_inter)
    '''

    #print("Mean: {:.4f}".format(np.mean([a,b,c])))

    details = "Divergence: {:.4f}\nIntersection: {:.4f}".format(orig_recon_kl_divergence[0],
                                                                orig_recon_hist_inter[0])

    plt.legend(loc='upper right')
    plt.title("Histogram Comparisons with Test Data: {}\nInfo: {}".format(
        test_loc, details))
    plt.show()


def dual_show_image(img_data_1, img_data_2):
    fig = plt.figure()
    ax_1 = fig.add_subplot(1, 2, 1)
    ax_1.set_title("First Image")
    ax_1.imshow((img_data_1/np.max(img_data_1) * 255).astype(np.uint8).T,
                interpolation='nearest', cmap="gray")

    ax_2 = fig.add_subplot(1, 2, 2)
    ax_2.set_title("Second Image")
    ax_2.imshow((img_data_2/np.max(img_data_2) * 255).astype(np.uint8).T,
                interpolation='nearest', cmap="gray")

    plt.show()


def show_image_diffs(img_data_1, img_data_2):
    fig = plt.figure()
    ax_1 = fig.add_subplot(1, 2, 1)
    ax_1.set_title("Original")
    ax_1.imshow((img_data_1/np.max(img_data_1) * 255).astype(np.uint8).T,
                interpolation='nearest', cmap="gray")

    ax_2 = fig.add_subplot(1, 2, 2)
    ax_2.set_title("Reconstructed")
    ax_2.imshow((img_data_2 * 255).astype(np.uint8).T,
                interpolation='nearest', cmap="gray")

    '''
    fig = plt.figure()
    ax_1 = fig.add_subplot(1,4,1)
    ax_1.set_title("Original")
    ax_1.imshow((img_data_1 * 255).astype(np.uint8).T, interpolation='nearest', cmap="gray")

    ax_2 = fig.add_subplot(1,4,2)
    ax_2.set_title("Reconstructed")
    ax_2.imshow((img_data_2 * 255).astype(np.uint8).T, interpolation='nearest', cmap="gray")

    ax_3 = fig.add_subplot(1,4,3)
    ax_3.set_title("Difference")
    diff = np.abs((img_data_1 * 255).T - (img_data_2 * 255).T).astype(np.uint8)
    ax_3.imshow(diff, interpolation='nearest', cmap="gray")

    ax_4 = fig.add_subplot(1,4,4)
    ax_4.set_title("Product")
    prod = ((img_data_1 * 255).T * (img_data_2 * 255).T).astype(np.uint8)
    ax_4.imshow(prod, interpolation='nearest', cmap="gray")

    fig.suptitle("Sum of Diff: {:.4f}\nMean of Diff: {:.4f}\nSum of Prod: {:.4f}\nMean of Prod: {:.4f}".format(np.sum(diff), np.mean(diff), np.sum(prod), np.mean(prod)))
    '''
    plt.show()
