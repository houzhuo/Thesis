# -*- coding: utf-8 -*-
import logging
# from numpy import *
import numpy  # for arrays, array broadcasting etc.
import numbers  # ?

from gensim import interfaces, utils, matutils
from itertools import chain
from scipy.special import gammaln, psi  # gamma function utils
from scipy.special import polygamma
#from six import xrange
import six

# log(sum(exp(x))) that tries to avoid overflow
# Compute the log of the sum of exponentials of input elements
try:
    # try importing from here if older scipy is installed
    from scipy.maxentropy import logsumexp
except ImportError:
    # maxentropy has been removed in recent releases, logsumexp now in misc
    from scipy.misc import logsumexp

logger = logging.getLogger('gensim.models.ldamodel')  # ?


def dirichlet_expectation(alpha):
    """
    For a vector 'theta~Dir(alpha)',compute 'E[log(theta)]'
    """
    if (len(alpha.shape) == 1):
        result = psi(alpha) - psi(numpy.sum(alpha))  # numpy.sum(a, axis=None, dtype=None, out=None, keepdims=False)
    else:
        result = psi(alpha) - psi(numpy.sum(alpha, 1))[:, numpy.newaxis]
        # psi:  The derivative of the logarithm of the gamma function evaluated at z (also called the digamma function)
    return result.astype(alpha.dtype)  # keep the same precision as input
    #  astype(type): returns a copy of the array converted to the specified type.
    #  dtype: Create a data type object

    #  >>> np.sum([[0, 1], [0, 5]])
    #  6
    #  >>> np.sum([[0, 1], [0, 5]], axis=0)
    #  array([0, 6])
    #  >>> np.sum([[0, 1], [0, 5]], axis=1)
    #  array([1, 5])

    # alpha = numpy.mat([[1],[2]])
    # print alpha.shape         # (2,1)
    # print len(alpha.shape)    # 2


def update_dir_prior(prior, N, logphat, rho):
    """
    Update a given using Newton's method
    """
    dprior = numpy.copy(prior)  # copy():copy a array
    gradf = N * (psi(numpy.sum(prior)) - psi(prior) + logphat)

    c = N * polygamma(1, numpy.sum(prior))  # This is the nth derivative of the digamma (psi) function.
    q = -N * polygamma(1, prior)

    b = numpy.sum(gradf / q) / (1 / c + numpy.sum(1 / q))

    dprior = -(gradf - b) / q

    if all(rho * dprior + prior) > 0:
        prior += rho * dprior
    else:
        logger.warning("update prior not positive")

    return prior


class LdaState(utils.SaveLoad):
    """
    Encapsulate information for distributed computation of LdaModel objects.
    Objects of this class are sent over the network, so try to keep them lean to
    reduce traffic.
    """

    def __init__(self, eta, shape):
        self.eta = eta
        self.sstats = numpy.zero(shape)
        self.numdocs = 0

    def reset(self):
        """
        prepare the state for a new EM iteration (reset sufficient stats)

        """
        self.sstats[:] = 0.0
        self.numdocs = 0

    def merge(self, other):
        """
        Merge the result of an E step from one node with that of another node
        (summing up sufficient statistics)

        The merging is trivial and after merging all the cluster nodes, we have the
        exact same result as if the computation was run on a single node(no approximation)

        """
        assert other is not None
        self.sstats += other.sstats
        self.numdocs += other.numdocs

    def blend(self, rhot, other, targetsize=None):
        """
        Given LdaState 'other', merge it with the current state. Strectch both to
        'targetsize' documents before merging, so that they are of comparable magitude.

        Merging is done by average weighting: in the extremes. 'rhot=0.0' means'other' is
        comletely ignored; 'rhot = 1.0'means 'self' is completely ignored

        this procedure corresponds to the gradient update form Hoffman  et al., algorithm 2 (eq. 14).
        """
        assert  other is not None
        if targetsize is None:
            targetsize = self.numdocs

        # stretch the current model's expected n*phi count to target size
        if self.numdocs == 0 or targetsize == self.numdocs:
            scale = 1.0
        else:
            scale = 1.0 * targetsize / self.numdocs
        self.sstats *= (1.0 - rhot) * scale

        #stretch the incoming n*phi counts to target size
        if other.numdocs == 0 or targetsize == other.numdocs:
            scale = 1.0
        else:
            logger.info("merging changes from %i documents into a model of %i documents", other.numdocs, targetsize)
            scale = 1.0 * targetsize / other.numdocs
        self.sstats += rhot * scale * other.sstats

        self.numdocs = targetsize

    def blend2(self, rhot, other, targetsize=None):
        """
        Alternative, more simple blend.
        """
        assert other is not None
        if targetsize is None:
            targetsize = self.numdocs

        # merge the 2 matrices by summing
        self.sstats += other.sstats
        self.numdocs = targetsize

    def get_lambda(self):
        return self.eta + self.sstats

    def get_Elogbeta(self):
        return dirichlet_expectation(self.get_lambda())

# endclass LdaState


class LdaModel(interfaces.TransformationABC):

    """
    The constructor estimates Latent Dirichlet Allocation model parameters based on a training corpus
    >>> lda = LdaModel(corpus, num_topics=10)

    You can then infer topic distributions on new, unseen documents, with
    >>> doc_lda = lda[doc_bow]

    The model can be updated(trained) with new documents via
    >>> lda.update(other_corpus)

    Model persistency is achieved through its 'load/save' methods.
    """

    def __init__(self, corpus=None, num_topics=100, id2word=None,
                 distributed=False, chunksize=2000, passes=1, update_every=1,
                 alpha='symmetric', eta=None, decay=0.5, offset=1.0,
                 eval_every=10, iterations=50, gamma_threshold=0.001,
                 minimum_probability=0.01):

        """
        If given, start training from the iterable 'corpus' straight away. If not given,
        the model is left untrained('update()')

        :param corpus:

        :param num_topics: is the number of requested latent topics to be extracted from
        the training corpus.

        :param id2word: is a mapping from word ids (integers) to words (strings). It is
        used to determine the vocabulary size, as well as for debugging and topic
        printing.

        :param eta and alpha: are hyperparameters that affect sparsity of the document-topic
        (theta) and topic-word(lambda) distributions. Both default to a symmetric
        1.0/num_topics prior

        :param alpha: can be set to an explicit array = prior of your choice. It also support
        special values of 'asymmetric' and 'auto': the former uses a fixed normalized asymmetric
        1.0/topicno prior, the latter learns an asymmetric prior directly from your data

        :param eta : can be a scalar for a symmetric prior over topic/word distributions, or a
        matrix of shape num_topics * num_words, which can be used to impose asymmetric priors over
        the word distribution on a per-topic basis. This may be useful if you want to seed certain
        topics with particular words by boosting the priors for those words. It also supports
        the special value 'auto', which learns an asymmetric prior directly from your data.

        :param distributed: Turn on 'distributed' ro force distributed computing
        (see the `web tutorial <http://radimrehurek.com/gensim/distributed.html>`_
        on how to set up a cluster of machines for gensim).

        :param eval_every: Calculate and log perplexity estimate from the latest mini-batch every
        `eval_every` model updates (setting this to 1 slows down training ~2x;
        default is 10 for better performance). Set to None to disable perplexity estimation.

        :param decay and offset:  decay` and `offset` parameters are the same as Kappa and Tau_0 in
        Hoffman et al, respectively.

        :param minimum_probability: controls filtering the topics returned for a documents(bow)

        Example:
        >>> lda = LdaModel(corpus, num_topics=100)  # train model
        >>> print(lda[doc_bow]) # get topic probability distribution for a document
        >>> lda.update(corpus2) # update the LDA model with additional documents
        >>> print(lda[doc_bow])
        >>> lda = LdaModel(corpus, num_topics=50, alpha='auto', eval_every=5)  # train asymmetric alpha from data
        """
        #store user-supplied parameters
        self.id2word = id2word
        if corpus is None and self.id2word is None:
            raise  ValueError('at least one of corpus/id2word must be specified, to establish input sapce dimensionality')

        if self.id2word is None:
            logger.warning("no word id mapping provided; initializing form corpus, assuming identity")
            self.id2word = utils.dict_from_corpus(corpus)
            self.num_terms = len(self.id2word)
        elif len(self.id2word) > 0:
            self.num_terms = 1 + max(self.id2word.keys())
        else:
            self.num_terms = 0
        if self.num_terms == 0:
            raise ValueError("cannot compute LDA over an empty collection (no terms)")

        self.distributed = bool(distributed)
        self.num_topics = int(num_topics)
        self.chunksize = chunksize
        self.decay = decay
        self.offset = offset
        self.minimum_probability = minimum_probability
        self.num_updates = 0

        self.iterations = iterations
        self.gamma_threshold = gamma_threshold

        self.passes = passes
        self.update_every = update_every
        self.eval_every = eval_every

        self.alpha, self.optimize_alpha = self.init_dir_prior(alpha, 'alpha')

        assert self.alpha.shape == (num_topics,), "Invalid alpha shape, Got shape %s, but expected (%d)" % (str(self.alpha.shape), num_topics)

        self.eta, self.optimize_eta = self.init_dir_prior(eta, 'eta')

        assert (self.eta.shape == num_topics, 1) or self.eta.shape == (num_topics, self.num_terms), (
            "Invalid eta shape. Got shape %s, but expected (%d, 1) or (%d, %d)" %
            (str(self.eta.shape),num_topics,num_topics, self.num_terms))


        #Initialize the variational distribution q(beta|lambda)
        self.state = LdaState(self.eta, (self.num_topics, self.num_terms))
        self.state.sstats = numpy.ramdom.gamma(100., 1 / 100./ (self.num_topics, self.num_terms))
        self.expElogbeta = numpy.exp(dirichlet_expectation(self.state.sstats))

        # if a training corpus was provided, start estimating the model right away
        if corpus is not None:
            use_numpy = self.dispatcher is not None
            self.update(corpus, chunks_as_numpy = use_numpy)


    def init_dir_prior(self, prior, name):
        if prior is None:
            prior = 'symmetric'

        is_auto = False

        if isinstance(prior, six.string_types):
            if prior == 'symmetric':
                logger.info("using symmetric %s at %s", name, 1.0/self.num_topics)
                init_prior = numpy.asanyarray([1.0 / self.num_topics for i in xrange(self.num_topics)])
            elif prior == 'asymmetric':
                init_prior = numpy.assrray([1.0 / (i + numpy.sqrt(self.num_topics)) for i in xrange(self.num_topics)])
                init_prior /= init_prior.sum()
                logger.info("using asymmetric %s %s", name, list(init_prior))
            elif prior == 'auto':
                is_auto = True
                init_prior = numpy.assarray([1.0 / self.num_topics for i in xrange(self.num_topics)])
                logger.info("using autotuned %s, starting with %s", name, list(init_prior))


    def __str__(self):
        return "LdaModel(num_terms=%s, num_topics=%s, decay=%s, chunksize=%s)" % \
               (self.num_terms, self.num_topics, self.decay, self.chunksize)

    def sync_state(self):
        self.expElogbeta = numpy.exp(self.state.get_Elogbeta())

    def clear(self):
        """Clear model state (free up some memory). Used in the distributed algo."""
        self.state = None
        self.Elogbeta = None

    def inference(self, chunk, collect_sstats=False):
        """
        Given a chunk of sparse document vectors, estimate gamma (parameters controlling the topic weights)
        for each document in the chunk.

        This function does not modify the model (=si read-only aka const). The whole input chunk of document
        is assumed to fit in RAM' chunking of a large corpus must be don earlier in the pipeline.

        if 'collect_sstats' is True, also collect sufficient statistics needed to update the model's
        topic-word distributions, and return a 2-tuple '(gamma, sstats)'. Otherwise, return '(gamma. None)'.
        'gamma' is of shape 'len(chunk) x self.num_topics'.

        Avoids computing the `phi` variational parameter directly using the
        optimization presented in **Lee, Seung: Algorithms for non-negative matrix factorization, NIPS 2001**.
        """

        try:
            _ = len(chunk)
        except:
            # convert iterators/generators to plain list, so we have len() etc.
            chunk = list(chunk)
        if len(chunk) > 1:
            logger.debug("performing inference on a chunk of %i documents", len(chunk))

        # Initialize the variational distribution q(theta|gamma) for the chunk
        gamma = numpy.random.gamma(100., 1. / 100., (len(chunk), self.num_topics))
        Elogtheta = dirichlet_expectation(gamma)
        expElogtheta = numpy.exp(Elogtheta)
        if collect_sstats:
            sstats = numpy.zeros_like(self.expElogbeta)
        else:
            sstats = None
        converged = 0

        # Now, for each document d update that document's gamma and phi
        # Inference code copied from Hoffman's `onlineldavb.py` (esp. the
        # Lee&Seung trick which speeds things up by an order of magnitude, compared
        # to Blei's original LDA-C code, cool!).
        for d, doc in enumerate(chunk):
            if doc and not isinstance(doc[0][0], six.integer_types):
                # make sure the term IDs are ints, otherwise numpy will get upset
                ids = [int(id) for id, _ in doc]
            else:
                ids = [id for id, _ in doc]
            cts = numpy.array([cnt for _, cnt in doc])
            gammad = gamma[d, :]
            Elogthetad = Elogtheta[d, :]
            expElogthetad = expElogtheta[d, :]
            expElogbetad = self.expElogbeta[:, ids]

            # The optimal phi_{dwk} is proportional to expElogthetad_k * expElogbetad_w.
            # phinorm is the normalizer.
            # TODO treat zeros explicitly, instead of adding 1e-100?













































if __name__ == '__main__':
    dirichlet_expectation()
