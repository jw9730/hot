from hot_pytorch.tests import test_set_util, test_dense_util, test_sparse_util
from hot_pytorch.tests import test_dense_linear, test_sparse_linear
from hot_pytorch.tests import test_dense_attn
from hot_pytorch.tests import test_sparse_attn


if __name__ == '__main__':
    try:
        test_set_util.test_masking()
        print('Set util test passed')
    except Exception as e:
        print(e)
        print('Set util test failed')
    try:
        test_dense_util.test_diag()
        test_dense_util.test_nondiag()
        test_dense_util.test_rotate()
        test_dense_util.test_batch()
        test_dense_util.test_batch_fn()
        print('Dense util test passed')
    except Exception as e:
        print(e)
        print('Dense util test failed')
    try:
        test_sparse_util.test_make_batch()
        test_sparse_util.test_transpose()
        test_sparse_util.test_diag()
        test_sparse_util.test_nondiag()
        print('Sparse util test passed')
    except Exception as e:
        print(e)
        print('Sparse util test failed')

    try:
        test_dense_linear.test_mask()
        test_dense_linear.test_1_0()
        test_dense_linear.test_1_1()
        test_dense_linear.test_1_2()
        test_dense_linear.test_2_0()
        test_dense_linear.test_2_1()
        test_dense_linear.test_2_2()
        print('Dense linear subroutine test passed')
    except Exception as e:
        print(e)
        print('Dense linear subroutine test failed')
    try:
        test_dense_linear.test_forward()
        test_dense_linear.test_pool()
        print('Dense linear forward test passed')
    except Exception as e:
        print(e)
        print('Dense linear forward test failed')
    try:
        test_dense_linear.test_backward()
        print('Dense linear backward test passed')
    except Exception as e:
        print(e)
        print('Dense linear backward test failed')

    try:
        test_dense_attn.test_attn()
        test_dense_attn.test_kernel_attn()
        print('Dense attn subroutine test passed')
    except Exception as e:
        print(e)
        print('Dense attn subroutine test failed')
    try:
        test_dense_attn.test_selfattn()
        print('Dense attn forward test passed')
    except Exception as e:
        print(e)
        print('Dense attn forward test failed')
    try:
        test_dense_attn.test_kernelselfattn()
        print('Dense kernel attn forward test passed')
    except Exception as e:
        print(e)
        print('Dense kernel attn forward test failed')
    try:
        test_dense_attn.test_backward()
        print('Dense attn backward test passed')
    except Exception as e:
        print(e)
        print('Dense attn backward test failed')

    try:
        test_sparse_linear.test_unique()
        test_sparse_linear.test_loop_mask()
        test_sparse_linear.test_1_0()
        test_sparse_linear.test_1_1()
        test_sparse_linear.test_2_0()
        test_sparse_linear.test_2_1()
        test_sparse_linear.test_2_2()
        print('Sparse linear subroutine test passed')
    except Exception as e:
        print(e)
        print('Sparse linear subroutine test failed')
    try:
        test_sparse_linear.test_forward()
        test_sparse_linear.test_pool()
        print('Sparse linear forward test passed')
    except Exception as e:
        print(e)
        print('Sparse linear forward test failed')
    try:
        test_sparse_linear.test_backward()
        print('Sparse linear backward test passed')
    except Exception as e:
        print(e)
        print('Sparse linear backward test failed')

    try:
        test_sparse_attn.test_attn()
        test_sparse_attn.test_kernel_attn()
        print('Sparse attn subroutine test passed')
    except Exception as e:
        print(e)
        print('Sparse attn subroutine test failed')
    try:
        test_sparse_attn.test_selfattn()
        print('Sparse attn forward test passed')
    except Exception as e:
        print(e)
        print('Sparse attn forward test failed')
    try:
        test_sparse_attn.test_kernelselfattn()
        print('Sparse kernel attn forward test passed')
    except Exception as e:
        print(e)
        print('Sparse kernel attn forward test failed')
    try:
        test_sparse_attn.test_backward()
        print('Sparse attn backward test passed')
    except Exception as e:
        print(e)
        print('Sparse attn backward test failed')
