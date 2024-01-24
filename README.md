# Zeta-Projekt

Das Zeta-Projekt implementiert einfache Funktionen in Python zur Berechnung von Burrows' Zeta, einer in der Computerlinguistik gebräuchlichen Maßeinheit zum Lesen von kontrastierenden Textkorpora.

In der Tatsache können Nutzer:innen anhand der einzelnen Funktionen Texte tokenisieren, lemmatisieren, den Tokens POS- und NER-Tags zuweisen, Listen von Tokens segmentieren, Stoppwörter und uninteressante Werte filtern, Ergebnisse sortieren. Alternativ kann das Skript auch als eigenständiges Programm ausgeführt werden, wobei die zur Berechnung der Zeta-Werte erforderlichen Daten direkt von den Benutzern:innen eingegeben werden.

## Dokumentation

Die Zeta-Projekt-Dokumentation wurde mit [Sphinx](https://www.sphinx-doc.org/en/master/index.html) unter Verwendung von *reStructuredText* erstellt und kann lokal abgerufen werden.


## Installation

Zeta-Projekt für Python >= 3.10 und alle seine Abhängigkeiten können nach den Anweisungen im [Python Packaging User Guide](https://packaging.python.org/en/latest/guides/section-install/) installiert werden. 

Zum Beispiel kann das Paket direkt aus seinem Git-Repository wie folgt installiert werden:

```bash
pip install zeta-project@git+https://github.com/silviapasc/zeta-project
```

Es wird außerdem empfohlen, eine virtuelle Umgebung einzurichten, in der das installierte Paket gespeichert wird.

## Lizenz

Dieses Projekt ist unter der [GPL-3-Lizenz](https://opensource.org/license/gpl-3-0/) lizenziert.
