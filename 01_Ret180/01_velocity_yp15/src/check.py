# Datasets size check ---------------------------------------------------------
#itr = iter(dataset_train)
#j = 0
#for i in range(n_samp_train):
#    example = next(itr)
#    j += 1
#
#try:
#    example = next(itr)
#except StopIteration:
#    print(f'Train set over: {j}')
#
#itr1 = iter(dataset_val)
#jj = 0
#for i in range(n_samp_valid):
#    example1 = next(itr1)
##    if np.any(np.isnan(example1[0].numpy())):
##         sys.exit(1)
##     elif np.any(np.isnan(example1[1][0].numpy())):
##         sys.exit(2)
##     elif np.any(np.isnan(example1[1][1].numpy())):
##         sys.exit(3)
##     elif np.any(np.isnan(example1[1][2].numpy())):
##         sys.exit(4)
#
#    jj += 1
#
#try:
#    example1 = next(itr1)
#except StopIteration:
#    print(f'Valid set over: {jj}')
#print(NAME)
#sys.exit(0)