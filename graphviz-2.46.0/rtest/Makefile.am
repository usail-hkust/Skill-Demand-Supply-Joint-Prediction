SUBDIRS = graphs linux.x86

check test rtest: $(top_builddir)/cmd/dot/dot_builtins $(top_builddir)/contrib/diffimg/diffimg
	./rtest.py

EXTRA_DIST = graphs nshare rtest.py strps.awk tests.txt tests_subset.txt test_regression.py
