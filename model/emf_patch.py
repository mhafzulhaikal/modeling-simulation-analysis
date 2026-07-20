import atexit
import os
import tempfile

import matplotlib.figure
import win32com.client

# Keep a reference to the original savefig
_original_savefig = matplotlib.figure.Figure.savefig
_ppt_app = None


def _get_ppt_app():
    global _ppt_app
    if _ppt_app is None:
        try:
            # Dispatch PowerPoint
            _ppt_app = win32com.client.Dispatch('PowerPoint.Application')
        except Exception as e:
            print(f'Warning: Could not start PowerPoint for EMF conversion: {e}')
    return _ppt_app


def _close_ppt_app():
    global _ppt_app
    if _ppt_app is not None:
        try:
            _ppt_app.Quit()
        except Exception:
            pass
        _ppt_app = None


atexit.register(_close_ppt_app)


def patched_savefig(self, fname, *args, **kwargs):
    if isinstance(fname, (str, os.PathLike)):
        fname_str = os.fspath(fname)
        if fname_str.lower().endswith('.emf'):
            # 1. Save to a temporary SVG file first (native vector format)
            fd, temp_svg = tempfile.mkstemp(suffix='.svg')
            os.close(fd)
            try:
                # Save as SVG (vector)
                _original_savefig(self, temp_svg, *args, **{**kwargs, 'format': 'svg'})

                # 2. Convert SVG to EMF using PowerPoint COM
                ppt = _get_ppt_app()
                if ppt is None:
                    raise RuntimeError('PowerPoint COM not available for EMF conversion')

                # Add a new blank presentation
                presentation = ppt.Presentations.Add(WithWindow=False)
                slide = presentation.Slides.Add(1, 12)  # 12 = ppLayoutBlank

                svg_abs = os.path.abspath(temp_svg)
                emf_abs = os.path.abspath(fname_str)

                # If destination file already exists, delete it first
                if os.path.exists(emf_abs):
                    try:
                        os.remove(emf_abs)
                    except Exception:
                        pass

                # Insert the SVG picture onto the slide
                shape = slide.Shapes.AddPicture(
                    FileName=svg_abs, LinkToFile=False, SaveWithDocument=True, Left=0, Top=0
                )

                # Export the shape to EMF format (2 = ppShapeFormatEMF)
                shape.Export(emf_abs, 2)
                presentation.Close()
                print(f'[OK] Figure saved to: {fname_str} (via PowerPoint EMF Conversion)')
                return
            except Exception as e:
                print(f'Error during EMF conversion for {fname_str}: {e}')
                print('Falling back to standard savefig format...')
                raise e
            finally:
                if os.path.exists(temp_svg):
                    try:
                        os.remove(temp_svg)
                    except Exception:
                        pass

    # Otherwise call the original savefig
    return _original_savefig(self, fname, *args, **kwargs)


# Apply patch
matplotlib.figure.Figure.savefig = patched_savefig
