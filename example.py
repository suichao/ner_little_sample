import json

from model import FSNERModel
from tokenizer_utils import FSNERTokenizerUtils
from utils import pretty_embed

query_texts = [
    "万通地产设计总监刘克峰；",
    "我最喜欢的小说是西游记"
]

# Each list in supports are the examples of one entity type
# Wrap entities around with [E] and [/E] in the examples.
# Each sentence should have only one pair of [E] ... [/E]

support_texts = {
        "place": [
            "[E]广州[/E]即将举行亚运会",
            "还记得[E]北京[/E]奥运会的开幕式吗",
            "知道[E]上海[/E]在哪里吗",
            "[E]佛山[/E]是著名的武术之乡",
            "[E]湛江[/E]是个美丽的海滨城市"
          ],
        "person": [
        "浙商银行企业信贷部[E]叶老桂[/E]博士则从另一个角度对五道门槛进行了解读。",
        "万通地产设计总监[E]刘克峰[/E]；",
        "方传柳实习生[E]王梦菲[/E]",
        "乐成中心的ceo[E]周东权[/E]先生",
        "接受锦旗的刑警[E]李济舟[/E]还原了案件经过。"
      ],
       "book": [
        "【吴海花】：我代表北京《[E]百姓地产[/E]》杂志感谢大家赶来参加今天《新老金融街争锋—",
        "沃霍尔手绘杰作“[E]大坎贝尔浓汤罐头[/E]”（蔬菜）是波普艺术的代表作",
        "花5000钱购买一本《[E]华胥梦天记[/E]》。",
        "银监会日前印发的《[E]关于进一步规范信用卡业务的通知[/E]》规定，",
        "中广网北京6月24日消息（记者张晶）据中国之声《[E]新闻纵横[/E]》7时42分报道，"
      ],
}

device = 'cpu'

tokenizer = FSNERTokenizerUtils("checkpoints/model/")
queries = tokenizer.tokenize(query_texts).to(device)
supports = tokenizer.tokenize(list(support_texts.values())).to(device)

model = FSNERModel("checkpoints/model/")
model.to(device)

p_starts, p_ends = model.predict(queries, supports)

# One can prepare supports once and reuse  multiple times with different queries
# ------------------------------------------------------------------------------
# start_token_embeddings, end_token_embeddings = model.prepare_supports(supports)
# p_starts, p_ends = model.predict(queries, start_token_embeddings=start_token_embeddings,
#                                  end_token_embeddings=end_token_embeddings)

output = tokenizer.extract_entity_from_scores(query_texts, queries, p_starts, p_ends,
                                              entity_keys=list(support_texts.keys()), thresh=0.80)

print(json.dumps(output, indent=2))

try:
    import spacy

    with open("result.html", "w") as f:
        f.write(pretty_embed(query_texts, output, list(support_texts.keys())))
except ImportError:
    print("Install spacy to output pretty embedding!")
