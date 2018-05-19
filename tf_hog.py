def tf_select_by_idx(a, idx, grayscale):
    if grayscale:
        return a[:,:,:,0]
    else:
        return tf.where(tf.equal(idx, 2), 
                         a[:,:,:,2], 
                         tf.where(tf.equal(idx, 1), 
                                   a[:,:,:,1], 
                                   a[:,:,:,0]))
    

def tf_hog_descriptor(images, cell_size = 8, block_size = 2, block_stride = 1, n_bins = 9,
                      grayscale = False):

    batch_size, height, width, depth = images.shape
    scale_factor = tf.constant(180 / n_bins, name="scale_factor", dtype=tf.float32)
    
    img = tf.constant(images, name="ImgBatch", dtype=tf.float32)

    if grayscale:
        img = tf.image.rgb_to_grayscale(img, name="ImgGray")

    # automatically padding height and width to valid size (multiples of cell size)
    if height % cell_size != 0 or width % cell_size != 0:
        height = height + (cell_size - (height % cell_size)) % cell_size
        width = width + (cell_size - (width % cell_size)) % cell_size
        img = tf.image.resize_image_with_crop_or_pad(img, height, width)
    
    # gradients
    grad = tf_deriv(img)
    g_x = grad[:,:,:,0::2]
    g_y = grad[:,:,:,1::2]
    
    # masking unwanted gradients of edge pixels
    mask_depth = 1 if grayscale else depth
    g_x_mask = np.ones((batch_size, height, width, mask_depth))
    g_y_mask = np.ones((batch_size, height, width, mask_depth))
    g_x_mask[:, :, (0, -1)] = 0
    g_y_mask[:, (0, -1)] = 0
    g_x_mask = tf.constant(g_x_mask, dtype=tf.float32)
    g_y_mask = tf.constant(g_y_mask, dtype=tf.float32)
    
    g_x = g_x*g_x_mask
    g_y = g_y*g_y_mask

    # maximum norm gradient selection
    g_norm = tf.sqrt(tf.square(g_x) + tf.square(g_y), "GradNorm")
    
    if not grayscale and depth != 1:
        # maximum norm gradient selection
        idx    = tf.argmax(g_norm, 3)
        g_norm = tf.expand_dims(tf_select_by_idx(g_norm, idx, grayscale), -1)
        g_x    = tf.expand_dims(tf_select_by_idx(g_x,    idx, grayscale), -1)
        g_y    = tf.expand_dims(tf_select_by_idx(g_y,    idx, grayscale), -1)

    g_dir = tf_rad2deg(tf.atan2(g_y, g_x)) % 180
    g_bin = tf.to_int32(g_dir / scale_factor, name="Bins")

    # cells partitioning
    cell_norm = tf.space_to_depth(g_norm, cell_size, name="GradCells")
    cell_bins = tf.space_to_depth(g_bin,  cell_size, name="BinsCells")

    # cells histograms
    hist = list()
    zero = tf.zeros(cell_bins.get_shape()) 
    for i in range(n_bins):
        mask = tf.equal(cell_bins, tf.constant(i, name="%i"%i))
        hist.append(tf.reduce_mean(tf.where(mask, cell_norm, zero), 3))
    hist = tf.transpose(tf.stack(hist), [1,2,3,0], name="Hist")

    # blocks partitioning
    block_hist = tf.extract_image_patches(hist, 
                                          ksizes  = [1, block_size, block_size, 1], 
                                          strides = [1, block_stride, block_stride, 1], 
                                          rates   = [1, 1, 1, 1], 
                                          padding = 'VALID',
                                          name    = "BlockHist")

    # block normalization
    block_hist = tf.nn.l2_normalize(block_hist, 3, epsilon=1.0)
    
    # HOG descriptor
    hog_descriptor = tf.reshape(block_hist, 
                                [int(block_hist.get_shape()[0]), 
                                 int(block_hist.get_shape()[1]) * \
                                 int(block_hist.get_shape()[2]) * \
                                 int(block_hist.get_shape()[3])], 
                                 name='HOGDescriptor')

    return hog_descriptor, block_hist, hist