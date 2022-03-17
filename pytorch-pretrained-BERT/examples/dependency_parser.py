from nltk.parse.stanford import StanfordDependencyParser

path_to_jar = "stanford-parser-full-2018-10-17/stanford-parser.jar"
path_to_models_jar = "stanford-parser-full-2018-10-17/stanford-parser-3.9.2-models.jar"  # noqa

dependency_parser = None


def _init_parser():
    global dependency_parser
    dependency_parser = StanfordDependencyParser(
        path_to_jar=path_to_jar,
        path_to_models_jar=path_to_models_jar
    )


def _get_parser():
    if dependency_parser is None:
        _init_parser()
    return dependency_parser


def parse_sents(sents):
    parses = _get_parser().parse_sents(sents)
    triples = [list(next(parse).triples()) for parse in parses]
    return triples


def parse_sent(sent):
    return list(next(_get_parser().parse(sent.split())).triples())
