import TextGen
import click
from nltk.tokenize import TweetTokenizer


@click.command()
@click.option(
    '--analyzer',
    default='word',
    help='Level of text generation (word or char levels)')
@click.option('--data', help='Path to the data folder')
@click.option(
    '--n_grams',
    default=4,
    help='Number of tokens on which our generation will base on')
@click.option('--gen_len', default=100, help='Prefered length of sequence')
def main(analyzer, data, n_grams, gen_len):
    tknzr = TweetTokenizer()
    text_gen = TextGen.TextGenerator(method='n_grams', analyzer=analyzer)
    prepared_text = text_gen.prepare_for_genetation(
        data_folder_path=data, tokenizer=tknzr.tokenize)
    text_gen.fit(prepared_text, n_grams=n_grams)
    if analyzer == 'word':
        click.echo(text_gen.generate(generate_len=gen_len, beautify=True))
    elif analyzer == 'char':
        click.echo(text_gen.generate(generate_len=gen_len))


if __name__ == '__main__':
    main()