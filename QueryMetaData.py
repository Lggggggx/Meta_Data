import numpy as np 
import copy

from sklearn.linear_model import LogisticRegression

from meta_data import DataSet, mate_data
from alipy.index import IndexCollection
from alipy.utils.misc import nlargestarg
from alipy.query_strategy.query_labels import QueryInstanceUncertainty, QueryRandom

class QueryMetaData():

    def __init__(self, X, y, metaregressor, label_inds_5=None, unlabel_inds_5=None, modelOutput_5=None):
        self.X = X
        self.y = y
        self.metaregressor = metaregressor
        dt = DataSet('au', X=X, y=y)
        self.distacne = dt.get_distance()
        _, self.cluster_center_index = dt.get_cluster_center()

        self.flag = False
        if label_inds_5 is not None:
            if unlabel_inds_5 is not None:
                if modelOutput_5 is not None:
                    self.label_inds_5 = label_inds_5
                    self.unlabel_inds_5 = unlabel_inds_5
                    self.modelOutput_5 = modelOutput_5
                    self.flag = True
        
        if self.flag is False:
            self.label_inds_5 = []
            self.unlabel_inds_5 = []
            self.modelOutput_5 = []

    def get_5_rouds(self, label_ind, unlabel_ind, Model, querystategy='random'):
        """
        label_ind: {list, np.ndarray, IndexCollection}
            The indexes of labeled samples.

        unlabel_ind: {list, np.ndarray, IndexCollection}
            The indexes of unlabeled samples.

        model: object, optional (default=None)
            Current classification model, should have the 'predict_proba' method for probabilistic output.
            If not provided, LogisticRegression with default parameters implemented by sklearn will be used.
        
        querystategy: str, default='uncertainty'
            In the first five rounds of active learning,choose to select the query strategy.
            Currently only supported uncertainty and random
        """
        assert(isinstance(label_ind, IndexCollection))
        assert(isinstance(unlabel_ind, IndexCollection))
        label_index = copy.deepcopy(label_ind)
        unlabel_index = copy.deepcopy(unlabel_ind)
        model = copy.deepcopy(Model)

        if querystategy =='uncertainty':
            un = QueryInstanceUncertainty(self.X, self.y)        
            for _ in range(5):
                select_ind = un.select(label_index, unlabel_index, model=model)
                label_index.update(select_ind)
                unlabel_index.difference_update(select_ind)
                self.label_inds_5.append(copy.deepcopy(label_index))
                self.unlabel_inds_5.append(copy.deepcopy(unlabel_index))
                model.fit(X=self.X[label_index.index, :], y=self.y[label_index.index])
                self.modelOutput_5.append(model.predict(self.X))

        elif querystategy =='random':
            random = QueryRandom(self.X, self.y)        
            for _ in range(5):
                select_ind = random.select(unlabel_index)
                label_index.update(select_ind)
                unlabel_index.difference_update(select_ind)
                self.label_inds_5.append(copy.deepcopy(label_index))
                self.unlabel_inds_5.append(copy.deepcopy(unlabel_index))
                model.fit(X=self.X[label_index.index, :], y=self.y[label_index.index])
                self.modelOutput_5.append(model.predict(self.X))

        self.flag = True

    def select(self, label_index, unlabel_index, model=None):
        """Select indexes from the unlabel_index for querying.

        Parameters
        ----------
        label_index: {list, np.ndarray, IndexCollection}
            The indexes of labeled samples.

        unlabel_index: {list, np.ndarray, IndexCollection}
            The indexes of unlabeled samples.

        model: object, optional (default=None)
            Current classification model, should have the 'predict_proba' method for probabilistic output.
            If not provided, LogisticRegression with default parameters implemented by sklearn will be used.

        Returns
        -------
        selected_idx: int
            The selected index.
        """
        if model is None:
            model = LogisticRegression()
        if self.flag is False:
            self.get_5_rouds(label_index, unlabel_index, model)

        label_ind = copy.deepcopy(self.label_inds_5[4])
        unlabel_ind = copy.deepcopy(self.unlabel_inds_5[4])
        metadata = self.cal_mate_data_Z(self.label_inds_5, self.unlabel_inds_5, self.modelOutput_5, model)
        metareg_perdict = self.metaregressor.predict(metadata)
        # print('len(metareg_perdict) ',len(metareg_perdict))
        select = np.argmax(metareg_perdict)
        # print('select ',select)
        # print('len(unlabel_ind)',len(unlabel_ind))
        select_ind = unlabel_ind[select]
        label_ind.update(select_ind)
        unlabel_ind.difference_update(select_ind)
        model.fit(X=self.X[label_index.index, :], y=self.y[label_index.index])

        # update the five rounds infor before
        del self.label_inds_5[0]
        del self.unlabel_inds_5[0]
        del self.modelOutput_5[0]
        
        self.label_inds_5.append(label_ind)
        self.unlabel_inds_5.append(unlabel_ind)
        self.modelOutput_5.append(model.predict(self.X))

        return select_ind

    def cal_mate_data_Z(self, label_inds, unlabel_inds, modelOutput, model):
        """calculate the designed mate data. 
        Parameters
        ----------
        label_inds: list
            index of labeling set, shape like [5, n_labeling_indexes]

        unlabel_inds: list
            index of unlabeling set, shape like [5, n_unlabeling_indexes]
        
        modelOutput: list
            each rounds model predition[5, n_samples]

        model: object, optional (default=None)
            Current classification model, should have the 'predict_proba' method for probabilistic output.
            If not provided, LogisticRegression with default parameters implemented by sklearn will be used.

        Returns
        -------
        metadata: 2D
            The meta data about the current model and dataset.[num_unlabel, 396(features)]
        """
        assert(len(label_inds)==5)
        assert(len(unlabel_inds)==5)
        assert(len(modelOutput)==5)

        metadata = None

        for j_sampelindex in unlabel_inds[4]:
            j_labelindex = copy.deepcopy(label_inds)
            j_unlabelindex = copy.deepcopy(unlabel_inds)
            jmodelOutput = copy.deepcopy(modelOutput)

            l_ind = copy.deepcopy(label_inds[4])
            u_ind = copy.deepcopy(unlabel_inds[4])

            j_u_ind = np.delete(u_ind, np.where(u_ind == j_sampelindex)[0])
            j_l_ind = np.r_[l_ind, j_sampelindex]

            j_labelindex.append(j_l_ind)
            j_unlabelindex.append(j_u_ind)

            model_j = copy.deepcopy(model)
            model_j.fit(self.X[j_l_ind], self.y[j_l_ind].ravel())
            # model`s predicted values continuous [-1, 1]
            # if modelname in ['RFR', 'DTR', 'ABR']:
            #     j_output = model_j.predict(X)
            # else:
            #     j_output = (model_j.predict_proba(X)[:, 1] - 0.5) * 2
            if hasattr(model_j, 'predict_proba'):
                j_output = (model_j.predict_proba(self.X)[:, 1] - 0.5) * 2
            else:
                j_output = model_j.predict(self.X)

            jmodelOutput.append(j_output)

            # calulate the designed mate_data Z
            j_meta_data = mate_data(self.X, self.y, self.distacne, self.cluster_center_index, j_labelindex, j_unlabelindex, jmodelOutput, j_sampelindex)

            if metadata is None:
                metadata = j_meta_data
            else:
                metadata = np.vstack((metadata, j_meta_data))

        return metadata