import scipy.optimize as optimize
from functools import partial
from tqdm import tqdm
from scipy.interpolate import interp1d


def signal_difference(signals, pixel_spectrum, weights):
    """Determine the total difference between the given signals and the pixel signal
    Parameters
    ==========

    signals: (np.array)
        all the signals array from the stxm dataset
    pixel_spectrum: (np.array)
        the signal coming from an indiviual signal
    weights: (np.array)
        all the weights array from the stxm dataset

    Returns
    =======
    The total squared difference between the pixel spectrum and the signal spectrums
    """
    # Turn spectrum into numpy array
    pixel_spectrum = np.asarray(pixel_spectrum)
    spectrum = []

    # For each weight multiply it by the corresponding signal
    for i, weight in enumerate(weights):
        spectrum.append(weight * signals[i])

    # Sum the weighted spectrum together
    spec = np.sum(spectrum, axis=0)

    # Return the difference of the actual spectrum with the caluclated one
    return np.sum((np.abs(spec - pixel_spectrum)) ** 2)


def weight_analysis(self, row, column):
    """Optimizes the weights for each signal for the pixel selected

    Parameters
    ==========

    self (XANESFrameset - ptycho)
        the ptycho XANESFrameset

    row (int)
        the row value you would like to calculate the weights for

    column (int)
        the column value you would like to calculate the weights for

    Returns
    =======
    A numpy array with a format of [(row, column), signal weights]
    """
    # Try to set up a progress bar
    try:
        self.pbar_val = self.pbar_val + 1
        self.pbar.update(self.pbar_val)
    except:
        pass

    # obtain the starting guess values as well as the signals
    starting_guess = self.starting_guess
    signals = self.signals_analysis

    # Obtain the spectrum for a User defined pixel
    pixel_spec = self.spectrum(pixel=(row, column))
    signal_dif_partial = partial(signal_difference, signals, pixel_spec)

    # Optimize the difference between the calculated spectrum and the actual spectrum
    output = optimize.minimize(signal_dif_partial, starting_guess).x
    weights = []
    for value in output:
        weights.append(value)

    # Return the row, column, weights, and True bool - gets the pbar working properly
    return [(row, column), weights, True]


def initialize_vectorize(num_rows, num_columns):
    """ Places the row and column number next to each other in an iterable array.
        Used for starmap multiprocessing pixels.

    Parameters
    ==========
    num_rows: (int)
        Total number of rows the user wants to calculate
    num_columns: (int)
        Total number of columns the user wants to calculate

    Returns
    =======
    An iterable array of pixel locations to be passed into the np.vectorize pixel analysis
    """

    # Set up the storing arrays
    master_row = []
    master_column = []

    # For each row and column defined by the User output their vales
    if isinstance(num_rows, int) and isinstance(num_columns, int):
        for i in range(0, num_rows):
            for j in range(0, num_columns):
                master_row.append(i)
                master_column.append(j)
    its = master_row, master_column

    return its


def weight_finder(self, signals):
    """Based on the initial signals determine the weight matrix for each signal

    Parameters
    ==========
    self (XANESFrameset - ptycho)
        The ptycho frameset

    signals (stxm caluclated signals)
        the return from calculate signals from the stxm dataset

    Returns
    =======
    a np.array of the weights of the initial signals associated with a given pixel
    """

    # Obtain the frame shape
    shapes = self.frame_shape()

    # Make all starting guesses 1
    starting_guess = np.ones((1, np.shape(signals)[0]))

    # Obtain the vectorized row and columns
    rows, columns = initialize_vectorize(shapes[0], shapes[1])

    # Set the signals and starting guesses
    self.signals_analysis = signals
    self.starting_guess = starting_guess

    # Vectorize the weight analysis
    vectorize_weight_analysis = np.vectorize(weight_analysis, excluded=['self', 'signals', 'starting_guess'])

    results = vectorize_weight_analysis(self, rows, columns)

    return results


def signals_crop(smaller_array_energies, larger_array_energies, larger_signals):
    """Crop the larger array of energies to match the same dimesions as the smaller array of energies

    Parameters
    ==========

    smaller_array_energies (XANESFrameset - pytcho)
        typically the pytcho frameset energies since those are the ones that are usually shorter in length

    larger_array_energies (XANESFrameset - stxm)
        typically the stxm frameset energies since those are the ones that are usually longer in length

    larger_signals (caluclate signals output)

    Returns
    =======
    The signals taken from the larger amount of energies cropped to the amount of energies given by the smaller
    dataset energies.
    """
    # Obtain the min and max values for the larger energy array
    min_val_large = np.floor(larger_array_energies.min())
    max_val_large = np.ceil(larger_array_energies.max())

    # Find the difference and make sure we will have an energy resolution of 75 points for every ev increase
    dif = (max_val_large - min_val_large) * 75
    dif = int(dif)

    # Create a new set of energies with a high energy resolution
    new_energies = np.linspace(min_val_large, max_val_large, num=dif, endpoint=True)

    out_signals = []

    # For each of the larger signals interpolate values for the smaller energy resolution.
    for l_sig in larger_signals:

        newfunc = interp1d(larger_array_energies, l_sig)
        ens = []
        for en in smaller_array_energies:
            ens.append(newfunc(en))
        out_signals.append(ens)

    # Return the new interpolated signals
    return np.asarray(out_signals)


def stxm_signals_4_ptycho(stxm_fs, ptycho_fs, n_components, method='nmf'):
    """Takes the signals determined from the sxdm and calculates their weights in the ptycho image

    Parameters
    ==========

    stxm_fs (XANESFrameset - stxm)
        the stxm frameset for the field of view

    ptycho_fs (XANESFrameset - ptycho)
        the ptycho framset for the field of view

    n_components (int)
        an integer of the number of components the user wants to calculate

    method (str)
        the method by which the signals are determined - see caluclate signals documentation

    Returns
    =======

    A stroable array of weights for the ptycho dataset, the stxm signals, the cropped stxm signals based
    on how long the ptycho energy list is

    """

    # Calculate the stxm signals and crop the stxm energies
    signals, weights = stxm_fs.calculate_signals(n_components=n_components, method=method)
    output = signals_crop(ptycho_fs.energies(), stxm_fs.energies(), signals)

    # Create a starting guess of weights of all ones
    ptycho_fs.starting_guess = np.ones((1, np.shape(signals)[0]))
    ptycho_fs.signals_analysis = output

    # Obtain the frameshape to vectorize the pixels
    r, c = ptycho_fs.frame_shape()[0], ptycho_fs.frame_shape()[1]

    # Vecorize the pixels
    row, column = initialize_vectorize(r, c)

    # Initiate the progressbar
    ptycho_fs.pbar_val = 0
    widgets = ['Progress: ', Percentage(), ' ', Bar(marker='-', left='[', right=']'),
               ' ', Timer(), '  ', ETA(), ' ', FileTransferSpeed()]  # see docs for other options
    ptycho_fs.pbar = ProgressBar(widgets=widgets, maxval=len(row) + 1)
    ptycho_fs.pbar.start()

    # Initialize the vectorization
    vectorize_weight_analysis = np.vectorize(weight_analysis, excluded=['self'])
    vectorized_results = vectorize_weight_analysis(ptycho_fs, row, column)

    # Return the vectorized results
    return readable(vectorized_results), signals, ptycho_fs.signals_analysis


def storeable(master_matrix):
    """Takes the np.vectorize output and makes it a readable and xanespy storeable array

    Parameters
    ==========
    master_matrix: nd.array
        the results from the readable(results) function

    Returns
    =======
    a nd.array that can be stored through xanespy
    """
    master = []
    iterations = np.shape(master_matrix)[2]
    for i in range(0, iterations):
        master.append(master_matrix[:, :, i])
    return np.asarray([master])


def readable(results):
    """Takes the np.vectorize output and makes it a readable and xanespy storeable array

    """
    shapes = np.add(results[-1][0], 1)

    master_matrix = [[0 for x in range(shapes[1])] for y in range(shapes[0])]
    for pixel in results:
        r, c = pixel[0]
        stored = pixel[1]
        master_matrix[r][c] = stored
    return storeable(np.asarray(master_matrix))


def readable_signal_maps(results):
    """Returns just the signal 2D map data
    """
    shapes = results[-1][0][-1][0]

    master_matrix = [[0 for x in range(shapes[1] + 1)] for y in range(shapes[0] + 1)]

    for pixel in results[0][0]:
        r, c = pixel[0]
        stored = pixel[1]
        master_matrix[r][c] = stored
    return storeable(np.asarray(master_matrix))


def merge_signals_stxm2ptycho(stxm_fs, ptycho_fs, n_components, method='nmf', mask_on=True):
    """Based on the stxm_fs, and ptycho_fs determine all the weights of each signal for each ptycho pixel

    Parameters
    ==========
    stxm_fs: XANESFrameset
        the sxtm_fs for the FOV the User is trying to upscale
    ptycho_fs: XANESFrameset
        the ptycho_fs for the FOV the User is trying to upscale
    n_components: int
        the number of components created from the calculate_signals() function
    method: str
        the method in which to create the signals
    mask_on: bool
        turns off the edge_mask for the signal analysis - helpful during the PCA analysis

    Returns
    =======
    The Ptycho weights for each STXM signal for each pixel, the STXM signals, and the cropped
    STXM signals used for fitting
    """
    # Obtain the stxm signals
    signals, weights = stxm_fs.calculate_signals(n_components=n_components, method=method, mask_on=mask_on)

    stxm_signals = np.asarray(signals)
    stxm_weights = np.asarray(weights)

    # Cropping the stxm signals to match the ptycho energy amount
    cropped_stxm_signals = signals_crop(ptycho_fs.energies(), stxm_fs.energies(), stxm_signals)

    # Calculating the ptycho weights of the signals for each pixel
    ptycho_weights = weight_finder(ptycho_fs, cropped_stxm_signals)

    return np.array([ptycho_weights, signals, cropped_stxm_signals])