from config import Config
from agent import Agent
from data_util import sample_data


def main():
    agent = Agent(Config)
    # agent.make_tsne_pic_for_directory()
    for i in range(2):
        agent.get_bounding_box()
    # agent.make_tsne_pic_for_directory()


if __name__ == '__main__':
    main()
