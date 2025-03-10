def kfold_split(dataset, k_splits):
    """
    Split the dataset into k folds
    Args:
        dataset: the list of sample features
        k_splits: the number of folds
    """
    assert len(dataset) > 0, 'Dataset is empty!'
    #cv_dataset_list = []  # [(trainset_1, testset_1), ..., (trainset_k, testset_k)]

    train_set = []
    val_set = []
    # chunk the dataset into k folds
    dataset_size = len(dataset)
    fold_size = dataset_size / float(k_splits)
    chunked_dataset = []
    last = 0.0
    split_counter = 1
    while split_counter <= k_splits:
        chunked_dataset.append(dataset[int(last):int(last + fold_size)])
        last += fold_size
        split_counter += 1
    assert len(chunked_dataset) == k_splits, 'The size of chunked_dataset should be same as k_splits!'

    for index in range(k_splits):
        val_set = chunked_dataset[index]
        #testset = chunked_dataset[index]
        #trainset = []
        for i in range(k_splits):
            if i == index:
                continue
            #train_set += chunked_dataset[i]
            train_set.extend(chunked_dataset[i])
        #train_set.extend(train_set)
        #train_test = (trainset, testset)
        #cv_dataset_list.append(train_test)
    return train_set,val_set

