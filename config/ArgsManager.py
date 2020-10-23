import argparse
import os

# Jupyterなどの勝手に変な引数が入る環境からの実行時にIGNORE_ARGS環境変数を設定することでargparseを完全に無視する
IGNORE_ARGS = False
if os.getenv("IGNORE_ARGS") is not None and os.getenv("IGNORE_ARGS") == "True":
    IGNORE_ARGS = True

# TODO: hydra便利そう
class ArgSetting:
    def __init__(self, default, key, type_class, domain_name=None):
        self.default = default
        self.key = key
        self.type = type_class
        self.domain_name = domain_name

    def __eq__(self, other):
        is_key_same = self.key.find(other.key) != -1 and len(self.key) == len(other.key)
        is_key_same = is_key_same and self.domain_name == other.domain_name
        is_type_same = self.type == other.type
        return is_key_same and is_type_same

    def __str__(self):
        if self.domain_name is None:
            domain_name = "GLOBAL"
        else:
            domain_name = self.domain_name
        return "Args(%s, %s, %s, %s)" % (domain_name, self.key, self.type, self.default)

    def __repr__(self):
        return self.__str__()

    def check_type(self, other):
        if self.key == other.key and self.type == other.type and self.domain_name != other.domain_name:
            assert False, "args type confliction. %s and %s" % (
            self.domain_name + "-" + self.key, other.domain_name + "-" + other.key)


class ArgsManager:
    def __init__(self):
        if IGNORE_ARGS:
            pass
        else:
            self.parser = argparse.ArgumentParser(conflict_handler='resolve')
        self.args = list()
        self.parsed_args = None
        self.finalized = False
        self.const_var_value_dict = None
        self.current_domain_name = None
        self.registered_arg_list = []

        self.GLOBAL_DOMAIN_NAME = "ARG_MANAGER_GLOBAL"

        self.DEFAULT_INT = -999999
        self.DEFAULT_FLOAT = -999999.0
        self.DEFAULT_STR = "DEFAULT_STR_BY_ARGUMENT_PARSER"
        self.DEFAULT_BOOL = "True"
        self.default_values = {
            str: self.DEFAULT_STR,
            int: self.DEFAULT_INT,
            float: self.DEFAULT_FLOAT,
            bool: self.DEFAULT_BOOL
        }

    def __str__(self):
        result = "ArgsManger\n"
        for arg in self.args:
            result += str(arg) + "\n"
        return result

    def set_domain(self, domain_name):
        self.current_domain_name = domain_name

    def merge_managers(self, args_managers):
        args = list()
        for args_manager in args_managers:
            for arg in args_manager.args:
                print("add", arg)
                args.append(arg)

        args_manager = ArgsManager()
        args_manager.args = args
        return args_manager

    def clone_arg_manager(self):
        args = self.args
        arg_manager = ArgsManager()
        arg_manager.args = args
        return arg_manager

    def load_const(self, domain_name=None):
        if not self.finalized:
            self.finalize_const()
        args_value_dict = self.parsed_args.__dict__
        result = dict()
        #
        for arg in self.args:
            if arg.domain_name in [self.GLOBAL_DOMAIN_NAME, domain_name]:
                key = arg.key.replace("-", "")
                KEY = key.upper()
                result[KEY] = args_value_dict[key]
                if arg.type == str:
                    result[KEY] = "'%s'" % result[KEY]
                elif arg.type == int:
                    result[KEY] = int(result[KEY])
                elif arg.type == float:
                    result[KEY] = float(result[KEY])
                elif arg.type == bool:
                    result[KEY] = bool(result[KEY])
        return result

    def finalize_const(self):
        if IGNORE_ARGS:
            return
        if not self.finalized:
            const_name_list = []
            registered_arg_list = []
            for arg_setting in self.args:
                const_name = arg_setting.key.replace('--', '').upper()
                if const_name not in const_name_list:
                    const_name_list.append(const_name)
                if arg_setting not in registered_arg_list:
                    registered_arg_list.append(arg_setting)
                    args_type = arg_setting.type if arg_setting.type != bool else str
                    print("add parse rule", arg_setting.key, args_type)
                    self.parser.add_argument(arg_setting.key, type=args_type,
                                             default=arg_setting.default)
            self.registered_arg_list = registered_arg_list
            self.parsed_args = self.parser.parse_args()
            self.finalized = True

    def overwritable(self, default, key, type_class, is_global=False):
        if IGNORE_ARGS:
            return default
        if is_global:
            setting = ArgSetting(default, key, type_class, self.GLOBAL_DOMAIN_NAME)
        else:
            setting = ArgSetting(default, key, type_class, self.current_domain_name)
        exist_flg = False
        for idx, arg in enumerate(self.args):
            if arg.key == setting.key:
                self.args[idx] = setting
                exist_flg = True
                break
        if not exist_flg:
            self.args.append(setting)
        return default

    def str(self, default, key, is_global=False):
        return self.overwritable(default, key, str, is_global=is_global)

    def int(self, default, key, is_global=False):
        return self.overwritable(default, key, int, is_global=is_global)

    def float(self, default, key, is_global=False):
        return self.overwritable(default, key, float, is_global=is_global)

    def bool(self, default, key, is_global=False):
        return self.overwritable(default, key, bool, is_global=is_global)


args = ArgsManager()
