#
In: utils.CRFtag_to_SCRFtag 
Op: add :
            if tag[0] == u'I':
                tags.append((beg, i, oldtag, tag[2:]))
                oldtag = tag[2:]
Reason: because of problem in the process of data. The program cannot process types correctly.