# How to Contribute

# Issues

* Please tag your issue with `bug`, `feature request`, or `question` to help us
  effectively respond.
* Please include the version of Fiddle you are running
  (run `pip list | grep fiddle-config`)
* Please provide the command line you ran as well as the log output.

# Pull Requests

We'd love to accept your patches and contributions to this project. There are
just a few small guidelines you need to follow.

## Contributor License Agreement

Contributions to this project must be accompanied by a Contributor License
Agreement. You (or your employer) retain the copyright to your contribution,
this simply gives us permission to use and redistribute your contributions as
part of the project. Head over to <https://cla.developers.google.com/> to see
your current agreements on file or to sign a new one.

You generally only need to submit a CLA once, so if you've already submitted one
(even if it was for a different project), you probably don't need to do it
again.

## Code reviews

All submissions, including submissions by project members, require review. We
use GitHub pull requests for this purpose. Consult
[GitHub Help](https://help.github.com/articles/about-pull-requests/) for more
information on using pull requests.

# Developing

Fiddle follows standard Python packaging best practices. The following steps
can be used to set up your development environment.

## One-time setup

### Git the source code

1. Fork the repository on GitHub by clicking on the **Fork** button on the
   [repository page](https://github.com/google/fiddle).
2. Clone the repository to your local machine: `git clone
https://github.com/$$YOUR_USERNAME$$/fiddle`.
3. Add the Fiddle repository as an upstream remote, so you can sync your
   changes: `git remote add upstream https://github.com/google/fiddle`.

### Python dependencies

> Note: if you're developing on a mac, you'll need xcode installed. (Required
> for PyType.)

1. Install Python (>= 3.7) locally.
2. Navigate to the directory containing your local clone of the git
   repository. (To ensure you're in the right directory, if you type
   `ls | grep CONTRIBUTING.md`, you should see this file listed.)
3. _Optional:_ Set up a virtual environment and activate it: `virtualenv
venv && source venv/bin/activate`.
4. Use `pip` to install your fork from source, including all testing
   dependencies: `pip install -e .[testing]`.
5. Test to ensure everything's working: `pytest fiddle`.

> Note: if you're developing on a mac, you'll need to exclude tests that
> depend on seqio (which is not available on macOS):
>
>      pytest \
>        --ignore fiddle/extensions/tf_test.py \
>        --ignore fiddle/extensions/seqio_test.py \
>        fiddle

## Make your changes

1. Create a branch where you will develop from:
   `git checkout -b name-of-change`.
2. Make whatever changes you'd like. You can run specific tests by passing a
   path to `pytest` (e.g. `pytest fiddle/config_test.py`).
3. Be sure to run all tests (`pytest fiddle`).
4. Ensure pytype passes for your changes (`pytype -j auto fiddle/config.py`).
   (Note: pytype is currently broken for a couple files in the repository.)
5. Commit the changes (provide a good description!), push to your fork, and
   send a PR following standard GitHub workflows.
