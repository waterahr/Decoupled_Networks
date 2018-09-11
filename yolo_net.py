import numpy as np
import tensorflow as tf
import yolo.config as cfg

slim = tf.contrib.slim

def leakyRelu(x, leak=0.2, name="LeakyRelu"):
    with tf.variable_scope(name):
        f1 = 0.5 * (1 + leak)
        f2 = 0.5 * (1 - leak)
        return f1 * x + f2 * tf.abs(x)

class YOLONet(object):

    def __init__(self, is_training=True):
        self.classes = cfg.CLASSES
        self.num_class = len(self.classes)
        self.image_size = cfg.IMAGE_SIZE
        self.cell_size = cfg.CELL_SIZE
        self.boxes_per_cell = cfg.BOXES_PER_CELL
        self.output_size = (self.cell_size * self.cell_size) *\
            (self.num_class + self.boxes_per_cell * 5)
        self.scale = 1.0 * self.image_size / self.cell_size
        self.boundary1 = self.cell_size * self.cell_size * self.num_class
        self.boundary2 = self.boundary1 +\
            self.cell_size * self.cell_size * self.boxes_per_cell

        self.object_scale = cfg.OBJECT_SCALE
        self.noobject_scale = cfg.NOOBJECT_SCALE
        self.class_scale = cfg.CLASS_SCALE
        self.coord_scale = cfg.COORD_SCALE
        self.angular_scale = cfg.ANGULAR_SCALE

        self.learning_rate = cfg.LEARNING_RATE
        self.batch_size = cfg.BATCH_SIZE
        self.alpha = cfg.ALPHA

        self.offset = np.transpose(np.reshape(np.array(
            [np.arange(self.cell_size)] * self.cell_size * self.boxes_per_cell),
            (self.boxes_per_cell, self.cell_size, self.cell_size)), (1, 2, 0))

        self.images = tf.placeholder(
            tf.float32, [None, self.image_size, self.image_size, 3],
            name='images')
        self.logits = self.build_network(
            self.images, num_outputs=self.output_size, alpha=self.alpha,
            is_training=is_training)

        if is_training:
            self.labels = tf.placeholder(
                tf.float32,
                [None, self.cell_size, self.cell_size, 5 + self.num_class])
            self.loss_layer(self.logits, self.labels)
            self.total_loss = tf.losses.get_total_loss()
            tf.summary.scalar('total_loss', self.total_loss)

    def build_network(self,
                      images,
                      num_outputs,
                      alpha,
                      keep_prob=0.5,
                      is_training=True,
                      scope='yolo'):
        with tf.variable_scope(scope):
            with slim.arg_scope(
                [slim.conv2d, slim.fully_connected],
                activation_fn=leaky_relu(alpha),
                weights_regularizer=slim.l2_regularizer(0.0005),
                weights_initializer=tf.truncated_normal_initializer(0.0, 0.01)
            ):
                net = tf.pad(
                    images, np.array([[0, 0], [3, 3], [3, 3], [0, 0]]),
                    name='pad_1')
                net = slim.conv2d(
                    net, 64, 7, 2, padding='VALID', scope='conv_2')
                net = slim.max_pool2d(net, 2, padding='SAME', scope='pool_3')
                net = slim.conv2d(net, 192, 3, scope='conv_4')
                net = slim.max_pool2d(net, 2, padding='SAME', scope='pool_5')
                net = slim.conv2d(net, 128, 1, scope='conv_6')
                net = slim.conv2d(net, 256, 3, scope='conv_7')
                net = slim.conv2d(net, 256, 1, scope='conv_8')
                net = slim.conv2d(net, 512, 3, scope='conv_9')
                net = slim.max_pool2d(net, 2, padding='SAME', scope='pool_10')
                net = slim.conv2d(net, 256, 1, scope='conv_11')
                net = slim.conv2d(net, 512, 3, scope='conv_12')
                net = slim.conv2d(net, 256, 1, scope='conv_13')
                net = slim.conv2d(net, 512, 3, scope='conv_14')
                net = slim.conv2d(net, 256, 1, scope='conv_15')
                net = slim.conv2d(net, 512, 3, scope='conv_16')
                net = slim.conv2d(net, 256, 1, scope='conv_17')
                net = slim.conv2d(net, 512, 3, scope='conv_18')
                net = slim.conv2d(net, 512, 1, scope='conv_19')
                net = slim.conv2d(net, 1024, 3, scope='conv_20')
                net = slim.max_pool2d(net, 2, padding='SAME', scope='pool_21')
                net = slim.conv2d(net, 512, 1, scope='conv_22')
                net = slim.conv2d(net, 1024, 3, scope='conv_23')
                net = slim.conv2d(net, 512, 1, scope='conv_24')
                net = slim.conv2d(net, 1024, 3, scope='conv_25')
                net = slim.conv2d(net, 1024, 3, scope='conv_26')
                net = tf.pad(
                    net, np.array([[0, 0], [1, 1], [1, 1], [0, 0]]),
                    name='pad_27')
                net = slim.conv2d(
                    net, 1024, 3, 2, padding='VALID', scope='conv_28')
                net = slim.conv2d(net, 1024, 3, scope='conv_29')
                net = slim.conv2d(net, 1024, 3, scope='conv_30')
                net = tf.transpose(net, [0, 3, 1, 2], name='trans_31')
                net = slim.flatten(net, scope='flat_32')
                net = slim.fully_connected(net, 512, scope='fc_33')
                net = slim.fully_connected(net, 4096, scope='fc_34')
                net = slim.dropout(
                    net, keep_prob=keep_prob, is_training=is_training,
                    scope='dropout_35')
                net = slim.fully_connected(
                    net, num_outputs, activation_fn=None, scope='fc_36')
        return net

    def calc_iou(self, boxes1, boxes2, scope='iou'):
        """calculate ious
        Args:
          boxes1: 5-D tensor [BATCH_SIZE, CELL_SIZE, CELL_SIZE, BOXES_PER_CELL, 4]  ====> (x_center, y_center, w, h)
          boxes2: 5-D tensor [BATCH_SIZE, CELL_SIZE, CELL_SIZE, BOXES_PER_CELL, 4] ===> (x_center, y_center, w, h)
        Return:
          iou: 4-D tensor [BATCH_SIZE, CELL_SIZE, CELL_SIZE, BOXES_PER_CELL]
        """
        with tf.variable_scope(scope):
            # transform (x_center, y_center, w, h) to (x1, y1, x2, y2)
            boxes1_t = tf.stack([boxes1[..., 0] - boxes1[..., 2] / 2.0,
                                 boxes1[..., 1] - boxes1[..., 3] / 2.0,
                                 boxes1[..., 0] + boxes1[..., 2] / 2.0,
                                 boxes1[..., 1] + boxes1[..., 3] / 2.0],
                                axis=-1)

            boxes2_t = tf.stack([boxes2[..., 0] - boxes2[..., 2] / 2.0,
                                 boxes2[..., 1] - boxes2[..., 3] / 2.0,
                                 boxes2[..., 0] + boxes2[..., 2] / 2.0,
                                 boxes2[..., 1] + boxes2[..., 3] / 2.0],
                                axis=-1)

            # calculate the left up point & right down point
            lu = tf.maximum(boxes1_t[..., :2], boxes2_t[..., :2])
            rd = tf.minimum(boxes1_t[..., 2:], boxes2_t[..., 2:])

            # intersection
            intersection = tf.maximum(0.0, rd - lu)
            inter_square = intersection[..., 0] * intersection[..., 1]

            # calculate the boxs1 square and boxs2 square
            square1 = boxes1[..., 2] * boxes1[..., 3]
            square2 = boxes2[..., 2] * boxes2[..., 3]

            union_square = tf.maximum(square1 + square2 - inter_square, 1e-10)

        return tf.clip_by_value(inter_square / union_square, 0.0, 1.0)

    def loss_layer(self, predicts, labels, scope='loss_layer'):
        with tf.variable_scope(scope):
            predict_classes = tf.reshape(
                predicts[:, :self.boundary1],
                [self.batch_size, self.cell_size, self.cell_size, self.num_class])
            predict_scales = tf.reshape(
                predicts[:, self.boundary1:self.boundary2],
                [self.batch_size, self.cell_size, self.cell_size, self.boxes_per_cell])
            predict_boxes = tf.reshape(
                predicts[:, self.boundary2:],
                [self.batch_size, self.cell_size, self.cell_size, self.boxes_per_cell, 4])
            #7*7*2*4(x, y, w, h)[label---绝对值(x, y, w, h);output---相对值(x, y)->cell,(w, h)->image]

            response = tf.reshape(
                labels[..., 0],
                [self.batch_size, self.cell_size, self.cell_size, 1])
            boxes = tf.reshape(
                labels[..., 1:5],
                [self.batch_size, self.cell_size, self.cell_size, 1, 4])
            boxes = tf.tile(
                boxes, [1, 1, 1, self.boxes_per_cell, 1]) / self.image_size
            classes = labels[..., 5:]

            offset = tf.reshape(
                tf.constant(self.offset, dtype=tf.float32),
                [1, self.cell_size, self.cell_size, self.boxes_per_cell])
            offset = tf.tile(offset, [self.batch_size, 1, 1, 1])
            offset_tran = tf.transpose(offset, (0, 2, 1, 3))
            predict_boxes_tran = tf.stack(
                [(predict_boxes[..., 0] + offset) / self.cell_size,
                 (predict_boxes[..., 1] + offset_tran) / self.cell_size,
                 tf.square(predict_boxes[..., 2]),
                 tf.square(predict_boxes[..., 3])], axis=-1)#相对于image而言的（x, y, w, h）百分比

            iou_predict_truth = self.calc_iou(predict_boxes_tran, boxes)#percent

            # calculate I tensor [BATCH_SIZE, CELL_SIZE, CELL_SIZE, BOXES_PER_CELL]
            object_mask = tf.reduce_max(iou_predict_truth, 3, keep_dims=True)
            object_mask = tf.cast(
                (iou_predict_truth >= object_mask), tf.float32) * response

            # calculate no_I tensor [CELL_SIZE, CELL_SIZE, BOXES_PER_CELL]
            noobject_mask = tf.ones_like(
                object_mask, dtype=tf.float32) - object_mask

            boxes_tran = tf.stack(
                [boxes[..., 0] * self.cell_size - offset,
                 boxes[..., 1] * self.cell_size - offset_tran,
                 tf.sqrt(boxes[..., 2]),
                 tf.sqrt(boxes[..., 3])], axis=-1)

            # class_loss
            class_delta = response * (predict_classes - classes)
            class_loss = tf.reduce_mean(
                tf.reduce_sum(tf.square(class_delta), axis=[1, 2, 3]),
                name='class_loss') * self.class_scale

            # object_loss
            object_delta = object_mask * (predict_scales - iou_predict_truth)
            object_loss = tf.reduce_mean(
                tf.reduce_sum(tf.square(object_delta), axis=[1, 2, 3]),
                name='object_loss') * self.object_scale

            # noobject_loss
            noobject_delta = noobject_mask * predict_scales
            noobject_loss = tf.reduce_mean(
                tf.reduce_sum(tf.square(noobject_delta), axis=[1, 2, 3]),
                name='noobject_loss') * self.noobject_scale

            # coord_loss
            coord_mask = tf.expand_dims(object_mask, 4)
            boxes_delta = coord_mask * (predict_boxes - boxes_tran)
            coord_loss = tf.reduce_mean(
                tf.reduce_sum(tf.square(boxes_delta), axis=[1, 2, 3, 4]),
                name='coord_loss') * self.coord_scale
            #print(coord_loss)
            

            # normal vector(angular) loss
            """
            norm_boxes = tf.reduce_sum(tf.square(boxes), axis=4)
            norm_predict_boxes_tran = tf.reduce_sum(tf.square(predict_boxes_tran), axis=4)
            normal_loss = tf.reduce_sum(predict_boxes_tran * boxes, axis=4) / (norm_boxes * norm_predict_boxes_tran)
            normal_loss = tf.reduce_sum(normal_loss)
            """
            """
            normal_loss = tf.Variable(0.0)
            #the points in the flat:(r, g, b) as the coord of th point]
            #(percent)
            #predict_boxes_tran: 5-D tensor [BATCH_SIZE, CELL_SIZE, CELL_SIZE, BOXES_PER_CELL, 4]  ====> (x_center, y_center, w, h)
            #boxes: 5-D tensor [BATCH_SIZE, CELL_SIZE, CELL_SIZE, BOXES_PER_CELL, 4] ===> (x_center, y_center, w, h)
            
            predict_boxes_t = tf.stack([predict_boxes_tran[..., 0] - predict_boxes_tran[..., 2] / 2.0,
                                 predict_boxes_tran[..., 1] - predict_boxes_tran[..., 3] / 2.0,
                                 predict_boxes_tran[..., 0] + predict_boxes_tran[..., 2] / 2.0,
                                 predict_boxes_tran[..., 1] + predict_boxes_tran[..., 3] / 2.0],
                                axis=-1) * self.image_size
            predict_boxes_t = tf.cast(predict_boxes_t, dtype = tf.int32)

            boxes_t = tf.stack([boxes[..., 0] - boxes[..., 2] / 2.0,
                                 boxes[..., 1] - boxes[..., 3] / 2.0,
                                 boxes[..., 0] + boxes[..., 2] / 2.0,
                                 boxes[..., 1] + boxes[..., 3] / 2.0],
                                axis=-1) * self.image_size
            boxes_t = tf.cast(boxes_t, dtype = tf.int32)
            
            
            #not consider the predict_prob
            #predict_flats = []
            #label_flats = []
            shape_for = boxes_t.get_shape()#(BATCH_SIZE, CELL_SIZE, CELL_SIZE, BOXES_PER_CELL, 4)
            #print("****")
            for i in range(shape_for[1]):
                for j in range(shape_for[2]):
                    for k in range(shape_for[3]):
                        for m in range(shape_for[0]):
                            wl = predict_boxes_t[m, i, j, k, 0]
                            wr = predict_boxes_t[m, i, j, k, 2]
                            ht = predict_boxes_t[m, i, j, k, 1]
                            hd = predict_boxes_t[m, i, j, k, 3]
                            if i+j+k+m == 0:
                                predict_flats = tf.concat([tf.reshape(self.images[m, wl:wr, ht:hd, :], (1, -1, 3))], axis=0)
                            else:
                                predict_flats = tf.concat([predict_flats, 
                                                  tf.reshape(self.images[m, wl:wr, ht:hd, :], (1, -1, 3))], axis=0)
                            wl = boxes_t[m, i, j, k, 0]
                            wr = boxes_t[m, i, j, k, 2]
                            ht = boxes_t[m, i, j, k, 1]
                            hd = boxes_t[m, i, j, k, 3]
                            if i+j+k+m == 0:
                                label_flats = tf.concat([tf.reshape(self.images[m, wl:wr, ht:hd, :], (1, -1, 3))], axis=0)
                            else:
                                label_flats = tf.concat([label_flats, 
                                                 tf.reshape(self.images[m, wl:wr, ht:hd, :], (1, -1, 3))], axis=0)
            #print("****")
            #print(np.shape(predict_flats))(1568,)
            #print(label_flats.get_shape())(1568, ?, 3)
            #print(predict_flats)
            #predict_flats = np.asarray(predict_flats)
            #label_flats = np.asarray(label_flats)
            def conv_tf(data):#data [?, 3]---tensor
                #print(data)
                dim1 = tf.reshape(data[:, 0], (1, -1))
                dim1 = tf.cast(dim1, dtype=tf.float32)
                dim2 = tf.reshape(data[:, 1], (1, -1))
                dim2 = tf.cast(dim2, dtype=tf.float32)
                dim3 = tf.reshape(data[:, 2], (1, -1))
                dim3 = tf.cast(dim3, dtype=tf.float32)
                #print(tf.reduce_mean(dim1))
                #print(dim1 - tf.reduce_mean(dim1))
                #print(tf.square((dim1 - tf.reduce_mean(dim1))))
                num = tf.size(data) - 1
                num = tf.cast(num, dtype=tf.float32)
                cov11 = tf.reduce_sum(tf.square((dim1 - tf.reduce_mean(dim1)))) / num
                cov22 = tf.reduce_sum(tf.square((dim2 - tf.reduce_mean(dim2)))) / num
                cov33 = tf.reduce_sum(tf.square((dim3 - tf.reduce_mean(dim3)))) / num
                cov12 = tf.reduce_sum((dim1 - tf.reduce_mean(dim1)) * (dim2 - tf.reduce_mean(dim2))) / num
                cov13 = tf.reduce_sum((dim1 - tf.reduce_mean(dim1)) * (dim3 - tf.reduce_mean(dim3))) / num
                cov23 = tf.reduce_sum((dim2 - tf.reduce_mean(dim2)) * (dim3 - tf.reduce_mean(dim3))) / num
                result_cov = tf.reshape([cov11, cov12, cov13], (1, -1))
                result_cov = tf.concat([result_cov, tf.reshape([cov12, cov22, cov23], (1, -1))], axis=0)
                result_cov = tf.concat([result_cov, tf.reshape([cov13, cov23, cov33], (1, -1))], axis=0)
                return result_cov
                
            for i in range(shape_for[0] * shape_for[1] * shape_for[2] * shape_for[3]):
                #predict_flat_cov = np.cov(np.asarray(predict_flats[i]))
                predict_flat_cov = conv_tf(predict_flats[i])
                #print(predict_flat_cov)
                #u, s, v_t = np.linalg.svd(predict_flat_cov)
                s, u, v = tf.svd(predict_flat_cov, full_matrices=True)
                
                #LookupError: No gradient defined for operation 'loss_layer/Svd_3134' (op type: Svd)
                
                #print(s)
                #print(u)
                #print(v)
                #vector_index = np.argmin(s)
                vector_index = tf.cast(tf.argmin(s), dtype=tf.int32)
                #print(vector_index)
                #predict_flat_normal = v_t[vector_index, :3]
                predict_flat_normal = v[:3, vector_index]
                #print(predict_flat_normal)
                label_flat_cov = conv_tf(label_flats[i])
                s, u, v = tf.svd(label_flat_cov, full_matrices=True)
                vector_index = tf.cast(tf.argmin(s), dtype=tf.int32)
                label_normal = v[:3, vector_index]
                normal_loss += tf.reduce_sum(predict_flat_normal * label_normal)
            #print(normal_loss) 
            """                

            tf.losses.add_loss(class_loss)
            tf.losses.add_loss(object_loss)
            tf.losses.add_loss(noobject_loss)
            tf.losses.add_loss(coord_loss)
            tf.losses.add_loss(normal_loss)

            tf.summary.scalar('class_loss', class_loss)
            tf.summary.scalar('object_loss', object_loss)
            tf.summary.scalar('noobject_loss', noobject_loss)
            tf.summary.scalar('coord_loss', coord_loss)
            #tf.summary.scalar('normal_loss', normal_loss)

            tf.summary.histogram('boxes_delta_x', boxes_delta[..., 0])
            tf.summary.histogram('boxes_delta_y', boxes_delta[..., 1])
            tf.summary.histogram('boxes_delta_w', boxes_delta[..., 2])
            tf.summary.histogram('boxes_delta_h', boxes_delta[..., 3])
            tf.summary.histogram('iou', iou_predict_truth)


def leaky_relu(alpha):
    def op(inputs):
        #return tf.nn.leaky_relu(inputs, alpha=alpha, name='leaky_relu')
        return leakyRelu(inputs, leak=alpha, name='leaky_relu')
    return op
