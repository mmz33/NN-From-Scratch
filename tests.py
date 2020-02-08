from config import Config
from engine import Engine
import unittest
import sys

# dummy json
_json = """
    {
        "task": "train",
        "network": {
            "layer1": {"class": "linear", "n_in": 10, "n_out": 20},
            "layer2": {"class": "softmax"}
        },
        "batch_size": 100,
        "lr": 0.3,
        "loss": "ce"
    }
    """


def test_parse_json():
    config = Config(_json)
    print('Parsed json:', config.json_dict)


def test_config_get_value():
    config = Config(_json)
    assert config.get_value('batch_size') == 100
    assert config.get_value('lr') == 0.3
    assert config.get_value('task') == 'train'
    assert config.get_value('epochs', 20) == 20


def test_engine_init():
    config = Config(_json)
    engine = Engine(config)
    engine.init_from_config()
    import inspect
    engine_members = inspect.getmembers(engine, lambda a: not inspect.isroutine(a))
    for member_name, value in engine_members:
        if not config.get_value(member_name):
            continue
        assert config.get_value(member_name) == value


def test_engine_init_network_from_config():
    config = Config(_json)
    engine = Engine(config)
    engine.init_from_config()
    engine.init_network_from_config()

    net = config.json_dict['network']
    idx = 1
    for module in engine.net_model.modules:
        if module.module_name:
            assert module.module_name == net['layer%i' % idx]['class']
            idx += 1


if __name__ == '__main__':
    if len(sys.argv) <= 1:
        for k, v in sorted(globals().items()):
            # functions that start with 'test_' are the ones to be executed and others are ignored (can be helper funcs)
            if k.startswith('test_'):
                print('-' * 40)
                print('Executing: %s' % k)
                try:
                    v()
                except unittest.SkipTest as ex:
                    print('SkipTest:', ex)
                print('Finished')
                print('-' * 40)
        print('All tests are done.')
    else:
        assert len(sys.argv) >= 2
        # if you want to test certain functions only
        for arg in sys.argv[1:]:
            print('Executing: %s' % arg)
            # execute any existing function
            if arg in globals():
                globals()[arg]()
            else:
                eval(arg)  # assume some python code is passed
