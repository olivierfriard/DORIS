
.. install and launch DORIS



**DORIS** can be downloaded from the `GiHub repository <https://github.com/olivierfriard/doris/releases>`_

All versions (Windows and Mac) are *portable*, they do not need to be installed on your system.

All previous versions of **DORIS** are available in the GitHub repository.



GNU/Linux
------------------------------------------------------------------------------------------------------------------------

DORIS run with Python >= 3.8

* Create a virtual environment (to isolate DORIS and dependencies from your system):


.. code-block:: bash

   python3 -m venv doris_venv


* Activate the virtual environment

.. code-block:: bash

   source doris_venv/bin/activate


The prompt should now be prefixed by **doris_venv**


* Install DORIS


.. code-block:: bash

   pip install doris-tracker


* Launch DORIS


.. code-block:: bash

   python3 -m doris


Microsoft-Windows
------------------------------------------------------------------------------------------------------------------------

Download the **doris-x.y.z-win64-portable.zip** file, unzip it and launch **start_doris.bat**.

Mac OS
------------------------------------------------------------------------------------------------------------------------

Download the **doris-x.y.z.dmg** image file, mount it and launch the **DORIS** program.





