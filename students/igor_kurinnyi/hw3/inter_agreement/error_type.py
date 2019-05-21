from enum import Enum


class ErrorType(Enum):
	Vt       = 'Verb tense'
	Vm       = 'Verb modal'
	V0       = 'Missing verb'
	Vform    = 'Verb form'
	SVA      = 'Subject-verb-agreement'
	ArtOrDet = 'Article or Determiner'
	Nn       = 'Noun number'
	Npos     = 'Noun possesive'
	Pform    = 'Pronoun form'
	Pref     = 'Pronoun reference'
	Prep     = 'Preposition'
	Wci      = 'Wrong collocation/idiom'
	Wa       = 'Acronyms'
	Wform    = 'Word form'
	Wtone    = 'Tone'
	Srun     = 'Runons, comma splice'
	Smod     = 'Dangling modifier'
	Spar     = 'Parallelism'
	Sfrag    = 'Fragment'
	Ssub     = 'Subordinate clause'
	WOinc    = 'Incorrect sentence form'
	WOadv    = 'Adverb/adjective position'
	Trans    = 'Link word/phrases'
	Mec      = 'Punctuation, capitalization, spelling, typos'
	Rloc     = 'Local redundancy'
	Cit      = 'Citation'
	Others   = 'Other errors'
	Um       = 'Unclear meaning (cannot be corrected)'
	noop     = 'No Operations'

	def __str__(self):
		return self.value
